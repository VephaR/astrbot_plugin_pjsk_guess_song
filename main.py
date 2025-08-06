import asyncio
import json
import random
import time
import os
import sqlite3
import re
import subprocess
import io
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from pilmoji import Pilmoji
import itertools
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from collections import defaultdict

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    AIOHTTP_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

try:
    from PIL.Image import Resampling
    LANCZOS = Resampling.LANCZOS
except ImportError:
    LANCZOS = 1


from astrbot.api import logger


try:
    from astrbot.api.event import filter, AstrMessageEvent
    from astrbot.api.star import Context, Star, register, StarTools
    import astrbot.api.message_components as Comp
    from astrbot.core.utils.session_waiter import session_waiter, SessionController
    from astrbot.api import AstrBotConfig
except ImportError:
    logger.error("Failed to import from astrbot.api, attempting fallback.")
    from astrbot.core.plugin import Plugin as Star, Context, register, filter, AstrMessageEvent  # type: ignore
    import astrbot.core.message_components as Comp  # type: ignore
    from astrbot.core.utils.session_waiter import session_waiter, SessionController  # type: ignore
    # Fallback for StarTools if it's missing in older versions
    class StarTools:
        @staticmethod
        def get_data_dir(plugin_name: str) -> Path:
            # 提供一个回退实现，模拟原始的 get_db_path 逻辑
            # 此路径是相对于包含 'plugins' 文件夹的目录
            return Path(__file__).parent.parent.parent.parent / 'data' / 'plugins_data' / plugin_name


# --- 插件元数据 ---
PLUGIN_NAME = "pjsk_guess_song"
PLUGIN_AUTHOR = "nichinichisou"
PLUGIN_DESCRIPTION = "PJSK猜歌插件"
PLUGIN_VERSION = "1.1.0"
PLUGIN_REPO_URL = "https://github.com/nichinichisou0609/astrbot_plugin_pjsk_guess_song"


# --- 数据库管理 ---
def get_db_path(context: Context, plugin_dir: Path) -> str:
    """获取插件的数据库文件路径"""
    plugin_data_dir = StarTools.get_data_dir(PLUGIN_NAME)
    os.makedirs(plugin_data_dir, exist_ok=True)
    return str(plugin_data_dir / "guess_song_data.db")


def init_db(db_path: str):
    """初始化数据库，确保表结构兼容。"""
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # 1. 创建核心表（如果不存在）
        # 注意：user_stats 的 user_id 没有主键，以兼容旧数据库和新的 _update_stats 逻辑
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_stats (
                user_id TEXT,
                user_name TEXT,
                score INTEGER DEFAULT 0,
                attempts INTEGER DEFAULT 0,
                correct_attempts INTEGER DEFAULT 0,
                daily_games_played INTEGER DEFAULT 0,
                last_played_date TEXT,
                daily_listen_songs INTEGER DEFAULT 0,
                last_listen_date TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mode_stats (
                mode TEXT PRIMARY KEY,
                attempts INTEGER DEFAULT 0,
                correct_attempts INTEGER DEFAULT 0
            )
        """)
        conn.commit()

        # 2. 安全地添加新功能可能需要的列（如果不存在）
        # 这是一个安全的"迁移"，可以防止因插件更新导致崩溃
        try:
            cursor.execute("PRAGMA table_info(user_stats);")
            existing_columns = {row[1] for row in cursor.fetchall()}
            columns_to_add = {
                "daily_games_played": "INTEGER DEFAULT 0",
                "last_played_date": "TEXT",
                "daily_listen_songs": "INTEGER DEFAULT 0",
                "last_listen_date": "TEXT",
                "correct_streak": "INTEGER DEFAULT 0",
                "max_correct_streak": "INTEGER DEFAULT 0",
                "group_scores": "TEXT DEFAULT '{}'"
            }
            
            for col, col_type in columns_to_add.items():
                if col not in existing_columns:
                    logger.info(f"向 user_stats 表添加新列以支持新功能: {col}")
                    cursor.execute(f"ALTER TABLE user_stats ADD COLUMN {col} {col_type};")
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"向 user_stats 添加列失败: {e}", exc_info=True)
            conn.rollback()


def load_song_data(resources_dir: Path) -> Optional[List[Dict]]:
    """加载 guess_song.json 数据"""
    try:
        songs_file = resources_dir / "guess_song.json"
        with open(songs_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        logger.error(f"加载歌曲数据失败: {e}. 请确保 'guess_song.json' 和 'musicVocals.json' 在 'resources' 目录中。")
        return None


def load_character_data(resources_dir: Path) -> Dict[str, Dict]:
    """
    加载角色数据，将角色ID映射到完整的角色信息字典。
    """
    characters_path = resources_dir / "characters.json"
    if not characters_path.exists():
        logger.warning(f"角色数据文件未找到: {characters_path}")
        return {}
    
    try:
        with open(characters_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 将角色ID映射到完整的角色信息
        char_map = {
            str(item.get("characterId")): item
            for item in data if item.get("characterId")
        }
        return char_map

    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"加载或解析角色数据失败: {e}")
        return {}


# --- 核心插件类 ---
@register(PLUGIN_NAME, PLUGIN_AUTHOR, PLUGIN_DESCRIPTION, PLUGIN_VERSION, PLUGIN_REPO_URL)
class GuessSongPlugin(Star):  # type: ignore
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.context = context
        self.config = config
        self.plugin_dir = Path(__file__).parent
        
        self.executor = ThreadPoolExecutor(max_workers=5)

        self.resources_dir = self.plugin_dir / "resources"
        self.output_dir = self.plugin_dir / "output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.db_path = get_db_path(context, self.plugin_dir)
        init_db(self.db_path)

        # 会话和游戏管理属性
        self.context.game_session_locks = getattr(self.context, "game_session_locks", {})
        self.context.active_game_sessions = getattr(self.context, "active_game_sessions", set())
        self.last_game_end_time = {}
        
        # 歌曲数据和列表
        self.song_data = load_song_data(self.resources_dir)
        self.character_data = load_character_data(self.resources_dir)
        self.abbr_to_char_id = {
            char_info['name'].lower(): int(char_id)
            for char_id, char_info in self.character_data.items() if char_info.get('name')
        }

        self.random_mode_decay_factor = self.config.get("random_mode_decay_factor", 0.75)
        
        self.available_songs = []
        self.available_vocalist_songs = []
        self.bundle_to_song_map = {}
        self.another_vocal_songs = []
        if self.song_data:
            for song_item in self.song_data:
                self.available_songs.append(song_item)
                if song_item.get("vocalists"):
                    self.available_vocalist_songs.append(song_item)
                if 'vocals' in song_item and song_item['vocals']:
                    # 填充包含 another_vocal 的歌曲列表
                    if any(v.get('musicVocalType') == 'another_vocal' for v in song_item['vocals']):
                        self.another_vocal_songs.append(song_item)
                    
                    for vocal in song_item['vocals']:
                        bundle_name = vocal.get('vocalAssetbundleName')
                        if bundle_name:
                            self.bundle_to_song_map[bundle_name] = song_item

        # 新增：预计算角色ID到ANOV歌曲的映射以优化性能
        self.char_id_to_anov_songs = defaultdict(list)
        for song in self.another_vocal_songs:
            processed_chars = set()
            for vocal in song.get('vocals', []):
                if vocal.get('musicVocalType') == 'another_vocal':
                    for char in vocal.get('characters', []):
                        char_id = char.get('characterId')
                        if char_id and char_id not in processed_chars:
                            self.char_id_to_anov_songs[char_id].append(song)
                            processed_chars.add(char_id)

        self.available_accompaniment_songs = []
        self.available_piano_songs = []
        self.available_bass_songs = []
        self.available_drums_songs = []
        self.available_vocals_songs = []
        self.available_piano_songs_bundles = set()
        self.preprocessed_tracks = {
            "accompaniment": set(), "bass_only": set(),
            "drums_only": set(), "vocals_only": set()
        }
        
        # --- 核心修复：直接从 self.config 读取插件配置 ---
        self.group_whitelist = self.config.get("group_whitelist", [])
        self.group_blacklist = self.config.get("group_blacklist", [])
        self.max_plays_per_day = self.config.get("max_plays_per_day", 10)
        self.max_listen_per_day = self.config.get("max_listen_per_day", 10)
        self.game_cooldown_seconds = self.config.get("game_cooldown_seconds", 30)
        self.lightweight_mode = self.config.get("lightweight_mode", False)

        # 远程资源和清单
        self.remote_manifest_url = self.config.get("remote_manifest_url")
        self.preprocessed_assets_url = self.config.get("preprocessed_assets_url")
        self.use_remote_resources = bool(self.remote_manifest_url and self.preprocessed_assets_url)
        self.manifest_data = {}
        self.http_session: Optional['aiohttp.ClientSession'] = None
        self.manifest_lock = asyncio.Lock()

        # 统计服务器配置（已修正为直接从 self.config 读取）
        self.api_key = self.config.get("stats_server_api_key")
        remote_url_base = self.config.get("remote_resource_url_base")
        if remote_url_base:
            try:
                parsed_url = urlparse(remote_url_base)
                # 假设统计服务器在同一主机上，但端口为 5000
                self.stats_server_url = f"{parsed_url.scheme}://{parsed_url.hostname}:5000"
            except Exception as e:
                logger.error(f"无法从 '{remote_url_base}' 解析统计服务器地址: {e}")
                self.stats_server_url = None
        else:
            self.stats_server_url = None

        self.game_effects = {
            # 稳定ID: {显示名}
            'speed_2x':   {'name': '2倍速'},
            'reverse':    {'name': '倒放'},
            'piano':      {'name': '钢琴'},
            'acc':        {'name': '伴奏'},
            'bass':       {'name': '纯贝斯'},
            'drums':      {'name': '纯鼓组'},
            'vocals':     {'name': '纯人声'},
        }

        # 游戏模式定义
        self.game_modes = {
            'normal': {'name': '普通', 'kwargs': {}, 'score': 1},
            '1': {'name': '2倍速', 'kwargs': {'speed_multiplier': 2.0}, 'score': 1},
            '2': {'name': '倒放', 'kwargs': {'reverse_audio': True}, 'score': 3},
            '3': {'name': 'AI-Assisted Twin Piano ver.', 'kwargs': {'melody_to_piano': True}, 'score': 2},
            '4': {'name': '纯伴奏', 'kwargs': {'play_preprocessed': 'accompaniment'}, 'score': 1},
            '5': {'name': '纯贝斯', 'kwargs': {'play_preprocessed': 'bass_only'}, 'score': 3},
            '6': {'name': '纯鼓组', 'kwargs': {'play_preprocessed': 'drums_only'}, 'score': 4},
            '7': {'name': '纯人声', 'kwargs': {'play_preprocessed': 'vocals_only'}, 'score': 1},
        }
        
        # 随机模式的效果
        self.base_effects = [
            {'name': '2倍速', 'kwargs': {'speed_multiplier': 2.0}, 'group': 'speed', 'score': 1},
            {'name': '倒放', 'kwargs': {'reverse_audio': True}, 'group': 'direction', 'score': 3},
        ]
        self.source_effects = [
            {'name': 'Twin Piano ver.', 'kwargs': {'melody_to_piano': True}, 'group': 'source', 'score': 2},
            {'name': '纯人声', 'kwargs': {'play_preprocessed': 'vocals_only'}, 'group': 'source', 'score': 1},
            {'name': '纯贝斯', 'kwargs': {'play_preprocessed': 'bass_only'}, 'group': 'source', 'score': 3},
            {'name': '纯鼓组', 'kwargs': {'play_preprocessed': 'drums_only'}, 'group': 'source', 'score': 4},
            {'name': '纯伴奏', 'kwargs': {'play_preprocessed': 'accompaniment'}, 'group': 'source', 'score': 1}
        ]

        self.listen_modes = {
            "piano": {"name": "钢琴", "list_attr": "available_piano_songs", "file_key": "piano"},
            "karaoke": {"name": "伴奏", "list_attr": "available_accompaniment_songs", "file_key": "accompaniment"},
            "vocals": {"name": "人声", "list_attr": "available_vocals_songs", "file_key": "vocals_only"},
            "bass": {"name": "贝斯", "list_attr": "available_bass_songs", "file_key": "bass_only"},
            "drums": {"name": "鼓组", "list_attr": "available_drums_songs", "file_key": "drums_only"},
        }
        self.mode_name_map = {}
        for key, value in self.game_modes.items():
            self.mode_name_map[key] = key
            self.mode_name_map[value['name'].lower()] = key
        for key, value in self.game_effects.items():
            self.mode_name_map[key] = key
            self.mode_name_map[value['name'].lower()] = key
        
        # 最终设置
        if self.song_data:
            self._populate_song_lists()
        else:
            logger.error("歌曲数据加载失败，插件部分功能可能无法使用。")

        self._cleanup_task = asyncio.create_task(self._periodic_cleanup_task())
        self._manifest_load_task = asyncio.create_task(self._load_remote_manifest())

    async def _get_session(self) -> Optional['aiohttp.ClientSession']:
        """延迟初始化并获取 aiohttp session"""
        if not AIOHTTP_AVAILABLE:
            return None
        if self.http_session is None or self.http_session.closed:
            self.http_session = aiohttp.ClientSession()
        return self.http_session

    def _load_local_manifest(self):
        """同步加载本地资源清单。"""
        logger.info("使用本地资源模式，开始扫描文件系统...")
        for mode in self.preprocessed_tracks.keys():
            mode_dir = self.resources_dir / mode
            if mode_dir.exists():
                for mp3_file in mode_dir.glob("*.mp3"):
                    self.preprocessed_tracks[mode].add(mp3_file.stem)
                logger.info(f"成功加载 {len(self.preprocessed_tracks[mode])} 个 '{mode}' 模式的本地音轨。")
        
        trimmed_mp3_dir = self.resources_dir / "songs_piano_trimmed_mp3"
        if trimmed_mp3_dir.exists():
            for mp3_path in trimmed_mp3_dir.glob("**/*.mp3"):
                self.available_piano_songs_bundles.add(mp3_path.parent.name)
            logger.info(f"成功加载 {len(self.available_piano_songs_bundles)} 个钢琴模式的本地音轨。")
        self._populate_song_lists() # 加载完后填充列表

    async def _load_remote_manifest(self):
        """异步加载远程资源清单。"""
        logger.info("使用远程资源模式，开始获取 manifest.json...")
        manifest_url = self._get_resource_path_or_url("manifest.json")
        if not manifest_url or not isinstance(manifest_url, str):
            logger.error("无法构建 manifest.json 的 URL。插件将无法使用预处理音轨模式。")
            return

        try:
            session = await self._get_session()
            if not session:
                logger.error("aiohttp session 不可用，无法获取远程 manifest。")
                return

            async with session.get(manifest_url, timeout=10) as response:
                response.raise_for_status()
                manifest_data = await response.json()
                
                for mode in self.preprocessed_tracks.keys():
                    self.preprocessed_tracks[mode] = set(manifest_data.get(mode, []))
                    logger.info(f"成功从 manifest 加载 {len(self.preprocessed_tracks[mode])} 个 '{mode}' 模式的音轨。")
                
                self.available_piano_songs_bundles = set(manifest_data.get("songs_piano_trimmed_mp3", []))
                logger.info(f"成功从 manifest 加载 {len(self.available_piano_songs_bundles)} 个钢琴模式的音轨。")
                self._populate_song_lists() # 加载完后填充列表

        except Exception as e:
            logger.error(f"获取或解析远程 manifest.json 失败: {e}。插件将无法使用预处理音轨模式。", exc_info=True)

    def _populate_song_lists(self):
        """根据已加载的音轨信息，填充可用的歌曲列表。"""
        if not self.song_data:
            return
            
        # 清空旧列表以支持重载
        self.available_accompaniment_songs.clear()
        self.available_vocals_songs.clear()
        self.available_bass_songs.clear()
        self.available_drums_songs.clear()
        self.available_piano_songs.clear()

        song_list_map = {
            'accompaniment': (self.available_accompaniment_songs, set()),
            'vocals_only': (self.available_vocals_songs, set()),
            'bass_only': (self.available_bass_songs, set()),
            'drums_only': (self.available_drums_songs, set()),
        }

        for mode, bundles in self.preprocessed_tracks.items():
            if mode in song_list_map:
                song_list, processed_ids = song_list_map[mode]
                for bundle_name in bundles:
                    if bundle_name in self.bundle_to_song_map:
                        song = self.bundle_to_song_map[bundle_name]
                        if song['id'] not in processed_ids:
                            song_list.append(song)
                            processed_ids.add(song['id'])
        
        piano_processed_ids = set()
        for bundle_name in self.available_piano_songs_bundles:
            if bundle_name in self.bundle_to_song_map:
                song = self.bundle_to_song_map[bundle_name]
                if song['id'] not in piano_processed_ids:
                    self.available_piano_songs.append(song)
                    piano_processed_ids.add(song['id'])
        logger.info(f"找到了 {len(self.available_piano_songs)} 首拥有预生成MP3的歌曲。")

    async def _periodic_cleanup_task(self):
        """每隔一小时自动清理一次 output 目录。"""
        cleanup_interval_seconds = 3600 # 1小时
        while True:
            await asyncio.sleep(cleanup_interval_seconds)
            logger.info("开始周期性清理 output 目录...")
            try:
                # 在线程池中执行IO密集型任务，避免阻塞事件循环
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self.executor, self._cleanup_output_dir)
            except Exception as e:
                logger.error(f"周期性清理任务失败: {e}", exc_info=True)

    def _get_resource_path_or_url(self, relative_path: str) -> Optional[Union[Path, str]]:
        """根据配置返回资源的本地Path对象或远程URL字符串。"""
        use_local = self.config.get("use_local_resources", True)
        if use_local:
            path = self.resources_dir / relative_path
            # 对于本地资源，我们可以直接检查其是否存在
            return path if path.exists() else None
        else:
            base_url = self.config.get("remote_resource_url_base", "").strip('/')
            if not base_url:
                logger.error("配置为使用远程资源，但 remote_resource_url_base 未设置。")
                return None
            # 对于远程资源，我们只构建URL，让调用方（如ffmpeg）处理连接
            # 使用posix风格的路径构建URL，以确保跨平台兼容性
            return f"{base_url}/{'/'.join(Path(relative_path).parts)}"

    async def _open_image(self, relative_path: str) -> Optional[Image.Image]:
        """打开一个资源图片，无论是本地路径还是远程URL。"""
        source = self._get_resource_path_or_url(relative_path)
        if not source:
            return None
        
        try:
            if isinstance(source, str) and source.startswith(('http://', 'https://')):
                session = await self._get_session()
                if not session:
                    logger.error("aiohttp session is not available.")
                    return None
                async with session.get(source) as response:
                    response.raise_for_status()
                    image_data = await response.read()
                    img = Image.open(io.BytesIO(image_data))
                    return img
            else:
                return Image.open(source)
        except Exception as e:
            logger.error(f"无法打开图片资源 {source}: {e}", exc_info=True)
            return None

    async def _get_duration_ms_ffprobe(self, file_path: Union[Path, str]) -> Optional[float]:
        """使用 ffprobe 高效获取音频时长，避免用 pydub 加载整个文件。"""
        command = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(file_path)
        ]
        try:
            loop = asyncio.get_running_loop()
            run_subprocess = partial(subprocess.run, command, capture_output=True, text=True, check=True, encoding='utf-8')
            result = await loop.run_in_executor(self.executor, run_subprocess)
            return float(result.stdout.strip()) * 1000
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as e:
            # FileNotFoundError: ffprobe not found
            # CalledProcessError: ffprobe returned non-zero exit code
            # ValueError: output is not a float
            logger.error(f"使用 ffprobe 获取时长失败 ({type(e).__name__}): {e}")
            return None

    def _is_group_allowed(self, event: AstrMessageEvent) -> bool:
        whitelist = self.config.get("group_whitelist", [])
        if not whitelist: return True
        # 显式转换为 bool 以匹配函数签名
        return bool(event.get_group_id() and str(event.get_group_id()) in whitelist)

    def get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _draw_options_image_sync(self, options: List[Dict], jacket_images: List[Optional[Image.Image]]) -> Optional[str]:
        """[辅助函数] 同步的选项图片绘制函数"""
        jacket_w, jacket_h = 128, 128
        padding = 15
        text_h = 50 
        cols, rows = 3, 4
        
        img_w = cols * jacket_w + (cols + 1) * padding
        img_h = rows * (jacket_h + text_h) + (rows + 1) * padding

        img = Image.new('RGBA', (img_w, img_h), (245, 245, 245, 255))
        
        try:
            font_path = str(self.resources_dir / "font.ttf")
            title_font = ImageFont.truetype(font_path, 16)
            num_font = ImageFont.truetype(font_path, 22) 
        except IOError:
            title_font = ImageFont.load_default()
            num_font = title_font

        draw = ImageDraw.Draw(img)

        for i, option in enumerate(options):
            jacket_img = jacket_images[i]
            if not jacket_img:
                continue

            row_idx, col_idx = i // cols, i % cols
            x = padding + col_idx * (jacket_w + padding)
            y = padding + row_idx * (jacket_h + text_h + padding)

            try:
                jacket = jacket_img.convert("RGBA").resize((jacket_w, jacket_h), LANCZOS)
                img.paste(jacket, (x, y), jacket)
                
                num_text = f"{i + 1}"
                circle_radius = 16
                circle_center = (x + circle_radius, y + circle_radius)
                draw.ellipse(
                    (circle_center[0] - circle_radius, circle_center[1] - circle_radius,
                     circle_center[0] + circle_radius, circle_center[1] + circle_radius),
                    fill=(0, 0, 0, 180)
                )
                
                with Pilmoji(img) as pilmoji_drawer:
                    pilmoji_drawer.text(circle_center, num_text, font=num_font, fill=(255, 255, 255), anchor="mm")

                title = option['title']
                if title_font.getbbox(title)[2] > jacket_w:
                     while title_font.getbbox(title + "...")[2] > jacket_w and len(title) > 1:
                         title = title[:-1]
                     title += "..."

                title_bbox = draw.textbbox((0, 0), title, font=title_font)
                title_w = title_bbox[2] - title_bbox[0]
                text_x = x + (jacket_w - title_w) / 2
                text_y = y + jacket_h + 8
                draw.text((text_x, text_y), title, font=title_font, fill=(30, 30, 50))

            except Exception as e:
                logger.error(f"处理歌曲封面失败: {option.get('title')}, 错误: {e}")
                continue

        img_path = self.output_dir / f"song_options_{int(time.time())}.png"
        img.save(img_path)
        return str(img_path)

    async def _create_options_image(self, options: List[Dict]) -> Optional[str]:
        """为12个歌曲选项创建一个3x4的图鉴"""
        if not options or len(options) != 12:
            return None

        # 1. 异步并发获取所有图片
        tasks = [self._open_image(f"music_jacket/{opt['jacketAssetbundleName']}.png") for opt in options]
        jacket_images = await asyncio.gather(*tasks)

        # 2. 在线程池中执行CPU密集的绘图操作
        loop = asyncio.get_running_loop()
        try:
            img_path = await loop.run_in_executor(
                self.executor, self._draw_options_image_sync, options, jacket_images
            )
            return img_path
        except Exception as e:
            logger.error(f"在executor中创建选项图片失败: {e}", exc_info=True)
            return None

    def _cleanup_output_dir(self, max_age_seconds: int = 3600):
        if not self.output_dir.exists(): return
        now = time.time()
        for filename in os.listdir(self.output_dir):
            file_path = self.output_dir / filename
            # 同时清理 png 和 wav/mp3
            if file_path.is_file() and (file_path.suffix in ['.png', '.wav', '.mp3']):
                if (now - file_path.stat().st_mtime) > max_age_seconds:
                    os.remove(file_path)
                    logger.info(f"已清理旧的输出文件: {filename}")

    async def start_new_game(self, **kwargs) -> Optional[Dict]:
        """
        准备一轮新游戏。
        该函数现在会智能选择处理路径：
        - 快速路径：对简单裁剪任务直接使用ffmpeg，性能更高。
        - 慢速路径：对需要变速、倒放等复杂效果的任务，使用pydub。
        """
        if not self.song_data or not PYDUB_AVAILABLE:
            logger.error("无法开始游戏: 歌曲数据未加载或pydub未安装。")
            return None

        # --- 步骤一：优先确定要玩的歌曲 ---
        song: Optional[Dict] = None
        preprocessed_mode = kwargs.get("play_preprocessed")
        is_piano_mode = kwargs.get("melody_to_piano", False)
        forced_song = kwargs.get("force_song_object")
        forced_vocal_version = kwargs.get("force_vocal_version")

        if forced_song:
            song = forced_song
        else:
            # 如果没有指定歌曲，则根据模式从可用池中随机选择
            if preprocessed_mode:
                available_bundles = self.preprocessed_tracks.get(preprocessed_mode, set())
                if not available_bundles:
                    logger.error(f"无法开始 {preprocessed_mode} 模式: 没有找到任何预处理的音轨文件。")
                    return None
                chosen_bundle = random.choice(list(available_bundles))
                song = self.bundle_to_song_map.get(chosen_bundle)
            elif is_piano_mode:
                if not self.available_piano_songs:
                    logger.error("无法开始钢琴模式: 没有找到任何预生成的钢琴曲。")
                    return None
                song = random.choice(self.available_piano_songs)
            else:
                song = random.choice(self.song_data)
        
        if not song:
            logger.error("在游戏准备的步骤一中未能确定歌曲。")
            return None

        # --- 步骤二：为已确定的歌曲查找对应的音频路径 ---
        audio_source: Optional[Union[Path, str]] = None
        audio_format = "mp3"

        if preprocessed_mode:
            possible_bundles = [
                v['vocalAssetbundleName'] for v in song.get('vocals', [])
                if v['vocalAssetbundleName'] in self.preprocessed_tracks.get(preprocessed_mode, set())
            ]
            if not possible_bundles:
                logger.error(f"歌曲 '{song.get('title')}' 没有适用于 '{preprocessed_mode}' 模式的可用音轨。")
                return None
            chosen_bundle = random.choice(possible_bundles)
            relative_path = f"{preprocessed_mode}/{chosen_bundle}.mp3"
            audio_source = self._get_resource_path_or_url(relative_path)
        
        elif is_piano_mode:
            # --- 修正：不再随机尝试所有vocal版本，而是精确匹配已知的可用钢琴音轨 ---
            # 1. 获取当前歌曲所有的 bundle name
            all_song_bundles = {v['vocalAssetbundleName'] for v in song.get('vocals', [])}
            
            # 2. 从已知的、拥有钢琴曲的bundle集合中，找出属于这首歌的
            valid_piano_bundles = list(all_song_bundles.intersection(self.available_piano_songs_bundles))
            
            if not valid_piano_bundles:
                # 理论上不应发生，因为选歌范围就是从available_piano_songs来的，这是一个安全保障
                logger.error(f"逻辑错误：歌曲 '{song.get('title')}' 在可用钢琴曲列表中，但找不到任何有效的音轨bundle。")
                return None
                
            # 3. 从真正有效的列表中随机选择一个来播放
            chosen_bundle = random.choice(valid_piano_bundles)
            relative_path = f"songs_piano_trimmed_mp3/{chosen_bundle}/{chosen_bundle}.mp3"
            audio_source = self._get_resource_path_or_url(relative_path)
            
            if not audio_source: # 最后的保险
                logger.error(f"歌曲 '{song.get('title')}' 没有适用于 '钢琴' 模式的可用音轨。")
                return None

        else: # 普通模式 or 猜歌手
            vocal_version = None
            if forced_vocal_version:
                vocal_version = forced_vocal_version
            else:
                if not song.get("vocals"):
                    logger.error(f"歌曲 '{song.get('title')}' 没有任何演唱版本信息。")
                    return None
                sekai_ver = next((v for v in song.get('vocals', []) if v.get('musicVocalType') == 'sekai'), None)
                vocal_version = sekai_ver if sekai_ver else random.choice(song.get("vocals", []))
            
            if vocal_version:
                bundle_name = vocal_version["vocalAssetbundleName"]
                relative_path = f"songs/{bundle_name}/{bundle_name}.mp3"
                audio_source = self._get_resource_path_or_url(relative_path)

        # --- 步骤三：验证路径并处理音频 ---
        if not audio_source:
            mode_name = preprocessed_mode or ('piano' if is_piano_mode else 'normal')
            song_title = song.get('title')
            logger.error(f"未能为歌曲 '{song_title}' 的 '{mode_name}' 模式找到有效的音频文件。")
            return None

        # --- 判断应使用快速路径还是慢速路径 ---
        is_bass_boost = preprocessed_mode == 'bass_only'
        has_speed_change = kwargs.get("speed_multiplier", 1.0) != 1.0
        has_reverse = kwargs.get("reverse_audio", False)
        has_band_pass = kwargs.get("band_pass")
        use_slow_path = is_bass_boost or has_speed_change or has_reverse or has_band_pass

        # --- 音频处理 ---
        loop = asyncio.get_running_loop()
        # 路径A: 快速路径 (直接使用ffmpeg，性能高)
        if not use_slow_path:
            try:
                total_duration_ms = await self._get_duration_ms_ffprobe(audio_source)
                if total_duration_ms is None:
                    raise ValueError("ffprobe failed or not found.")

                target_duration_ms = int(self.config.get("clip_duration_seconds", 10) * 1000)
                if preprocessed_mode in ["drums_only", "bass_only"]:
                    target_duration_ms *= 2

                source_duration_ms = target_duration_ms
                
                if source_duration_ms >= total_duration_ms:
                    start_ms = 0
                    duration_to_clip_ms = total_duration_ms
                else:
                    start_range_min = 0
                    if not preprocessed_mode and not is_piano_mode:
                        start_range_min = int(song.get("fillerSec", 0) * 1000)
                    
                    start_range_max = int(total_duration_ms - source_duration_ms)
                    start_ms = random.randint(start_range_min, start_range_max) if start_range_min < start_range_max else start_range_min
                    duration_to_clip_ms = source_duration_ms

                clip_path_obj = self.output_dir / f"clip_{int(time.time())}.mp3"
                command = [
                    'ffmpeg',
                    '-ss', str(start_ms / 1000.0),
                    '-i', str(audio_source),
                    '-t', str(duration_to_clip_ms / 1000.0),
                    '-c', 'copy',
                    '-y', str(clip_path_obj)
                ]
                
                run_subprocess = partial(subprocess.run, command, capture_output=True, text=True, check=True, encoding='utf-8')
                result = await loop.run_in_executor(self.executor, run_subprocess)

                if result.returncode != 0:
                    raise RuntimeError(f"ffmpeg clipping failed: {result.stderr}")
                
                # 修正：优先使用随机模式的名称
                mode = "normal"
                if kwargs.get("random_mode_name"):
                    mode = kwargs["random_mode_name"]
                elif kwargs.get('game_type') == 'guess_song_vocalist':
                    mode = 'guess_song_vocalist'
                elif preprocessed_mode:
                    mode = preprocessed_mode
                elif is_piano_mode:
                    mode = "melody_to_piano"
                elif kwargs.get('reverse_audio'):
                # 对应 game_modes 中的模式 '2'
                    mode = 'reverse_audio'
                elif kwargs.get('speed_multiplier'):
                # 对应 game_modes 中的模式 '1'
                    mode = 'speed_multiplier'
                
                return {"song": song, "clip_path": str(clip_path_obj), "score": kwargs.get("score", 1), "mode": mode, "game_type": kwargs.get('game_type')}

            except Exception as e:
                logger.warning(f"快速路径处理失败: {e}. 将回退到 pydub 慢速路径。")
        
        # 路径B: 慢速路径 (使用pydub，兼容复杂效果)
        try:
            audio_data: Union[str, Path, io.BytesIO]
            if isinstance(audio_source, str) and audio_source.startswith(('http://', 'https://')):
                session = await self._get_session()
                if not session:
                    logger.error("aiohttp session不可用，无法下载远程音频。")
                    return None
                async with session.get(audio_source) as response:
                    response.raise_for_status()
                    audio_data = io.BytesIO(await response.read())
            else:
                audio_data = audio_source

            pydub_kwargs = {
                "preprocessed_mode": preprocessed_mode,
                "target_duration_seconds": self.config.get("clip_duration_seconds", 10),
                "speed_multiplier": kwargs.get("speed_multiplier", 1.0),
                "reverse_audio": kwargs.get("reverse_audio", False),
                "band_pass": kwargs.get("band_pass"),
                "is_piano_mode": is_piano_mode,
                "song_filler_sec": song.get("fillerSec", 0)
            }
            
            # 将pydub的CPU密集型操作放入线程池
            clip = await loop.run_in_executor(
                self.executor,
                self._process_audio_with_pydub,
                audio_data,
                audio_format,
                pydub_kwargs
            )

            if clip is None:
                raise RuntimeError("pydub audio processing failed.")

            # 修正：优先使用随机模式的名称
            mode = "normal"
            if kwargs.get("random_mode_name"):
                mode = kwargs["random_mode_name"]
            elif kwargs.get('game_type') == 'guess_song_vocalist':
                mode = 'guess_song_vocalist'
            elif preprocessed_mode:
                mode = preprocessed_mode
            elif is_piano_mode:
                mode = "melody_to_piano"
            elif kwargs.get('reverse_audio'):
                # 对应 game_modes 中的模式 '2'
                mode = 'reverse_audio'
            elif kwargs.get('speed_multiplier'):
                # 对应 game_modes 中的模式 '1'
                mode = 'speed_multiplier'
            
            clip_path = self.output_dir / f"clip_{int(time.time())}.mp3"
            clip.export(clip_path, format="mp3", bitrate="128k")

            return {"song": song, "clip_path": str(clip_path), "score": kwargs.get("score", 1), "mode": mode, "game_type": kwargs.get('game_type')}

        except Exception as e:
            logger.error(f"慢速路径 (pydub) 处理音频文件 {audio_source} 时失败: {e}", exc_info=True)
            return None

    def _process_audio_with_pydub(self, audio_data: Union[str, Path, io.BytesIO], audio_format: str, options: dict) -> Optional['AudioSegment']:
        """[辅助函数] 在线程池中执行的同步pydub处理逻辑"""
        try:
            audio = AudioSegment.from_file(audio_data, format=audio_format)

            preprocessed_mode = options.get("preprocessed_mode")
            if preprocessed_mode == "bass_only":
                audio += 6

            target_duration_ms = int(options.get("target_duration_seconds", 10) * 1000)
            if preprocessed_mode in ["bass_only", "drums_only"]:
                target_duration_ms *= 2
            
            speed_multiplier = options.get("speed_multiplier", 1.0)
            source_duration_ms = int(target_duration_ms * speed_multiplier)
            total_duration_ms = len(audio)
            
            if source_duration_ms >= total_duration_ms:
                clip_segment = audio
            else:
                start_range_min = 0
                if not preprocessed_mode and not options.get("is_piano_mode"):
                    start_range_min = int(options.get("song_filler_sec", 0) * 1000)
                
                start_range_max = total_duration_ms - source_duration_ms
                start_ms = random.randint(start_range_min, start_range_max) if start_range_min < start_range_max else start_range_min
                end_ms = start_ms + source_duration_ms
                clip_segment = audio[start_ms:end_ms]

            clip = clip_segment
            
            if speed_multiplier != 1.0:
                clip = clip._spawn(clip.raw_data, overrides={'frame_rate': int(clip.frame_rate * speed_multiplier)})
            if options.get("reverse_audio", False):
                clip = clip.reverse()
            
            band_pass = options.get("band_pass")
            if band_pass and isinstance(band_pass, tuple) and len(band_pass) == 2:
                low_freq, high_freq = band_pass
                clip = clip.high_pass_filter(low_freq).low_pass_filter(high_freq) + 6
            
            return clip
        except Exception as e:
            logger.error(f"Pydub processing in executor failed: {e}", exc_info=True)
            return None

    async def _check_game_start_conditions(self, event: AstrMessageEvent) -> Tuple[bool, Optional[str]]:
        """检查是否可以开始新游戏，返回(布尔值, 提示信息)"""
        if not self._is_group_allowed(event): 
            return False, None
            
        session_id = event.unified_msg_origin
        cooldown = self.config.get("game_cooldown_seconds", 60)
        debug_mode = self.config.get("debug_mode", False)
        
        if not debug_mode and time.time() - self.last_game_end_time.get(session_id, 0) < cooldown:
            remaining_time = cooldown - (time.time() - self.last_game_end_time.get(session_id, 0))
            time_display = f"{remaining_time:.3f}" if remaining_time < 1 else str(int(remaining_time))
            return False, f"嗯......休息 {time_display} 秒再玩吧......"
        
        if session_id in self.context.active_game_sessions:
            return False, "......有一个正在进行的游戏了呢。"
        
        loop = asyncio.get_running_loop()
        can_play = await loop.run_in_executor(self.executor, self._can_play, event.get_sender_id())
        if not debug_mode and not can_play:
            limit = self.config.get('daily_play_limit', 15)
            return False, f"......你今天的游戏次数已达上限（{limit}次），请明天再来吧......"
            
        return True, None

    async def _run_game_session(self, event: AstrMessageEvent, game_data: Dict, intro_messages: List, answer_reveal_messages: List):
        """（已重构）统一的游戏会话执行器，包含简化的统计逻辑。"""
        session_id = event.unified_msg_origin
        debug_mode = self.config.get("debug_mode", False)
        timeout_seconds = self.config.get("answer_timeout", 30)
        correct_players = {}
        first_correct_answer_time = 0
        game_ended_by_attempts = False
        guessed_users = set()
        guess_attempts_count = 0
        max_guess_attempts = self.config.get("max_guess_attempts", 10)
        game_results_to_log = []

        try:
            await event.send(event.chain_result([Comp.Record(file=game_data["clip_path"])]))
            await event.send(event.chain_result(intro_messages))

            if debug_mode:
                logger.info("[猜歌插件] 调试模式已启用，立即显示答案")
                await event.send(event.chain_result(answer_reveal_messages))
                return
        except Exception as e:
            logger.error(f"发送消息失败: {e}. 游戏中断。", exc_info=True)
            return
        finally:
            if debug_mode:
                if session_id in self.context.active_game_sessions:
                    self.context.active_game_sessions.remove(session_id)
                self.last_game_end_time[session_id] = time.time()

        @session_waiter(timeout=timeout_seconds)
        async def unified_waiter(controller: SessionController, answer_event: AstrMessageEvent):
            nonlocal guess_attempts_count, correct_players, game_ended_by_attempts, first_correct_answer_time, guessed_users
            
            user_id = answer_event.get_sender_id()
            user_name = answer_event.get_sender_name()
            answer_text = answer_event.message_str.strip()
            
            if not answer_text.isdigit():
                return
            
            if user_id in guessed_users:
                return
            guessed_users.add(user_id)
            
            guess_attempts_count += 1
            
            is_correct = False
            try:
                answer_num = int(answer_text)
                if 1 <= answer_num <= game_data.get("num_options", 12):
                    if answer_num == game_data['correct_answer_num']:
                        is_correct = True
            except ValueError:
                pass

            score_to_add = 0
            can_score = False
            if is_correct:
                bonus_time = self.config.get("bonus_time_after_first_answer", 5)
                is_first_correct_answer = (first_correct_answer_time == 0)
                can_score = is_first_correct_answer or (bonus_time > 0 and (time.time() - first_correct_answer_time) <= bonus_time)
                if can_score:
                    score_to_add = game_data.get("score", 1)

            if game_data.get('game_type', '').startswith('guess_song'):
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self.executor, self._update_stats, session_id, user_id, user_name, score_to_add, is_correct)
                if score_to_add > 0:
                     asyncio.create_task(self._api_update_score(user_id, user_name, score_to_add))

                # 修正：使用 game_data['mode'] 以确保随机模式的稳定ID被正确记录
                await loop.run_in_executor(self.executor, self._update_mode_stats, game_data['mode'], is_correct)
                
                # 修改：将游戏结果暂存，游戏结束后统一发送
                game_results_to_log.append({
                    "game_type": game_data.get('game_type', 'guess_song'),
                    "game_mode": game_data['mode'], # 修正：同上
                    "user_id": user_id,
                    "user_name": user_name,
                    "is_correct": is_correct,
                    "score_awarded": score_to_add, # 修正：字段名以匹配服务器
                    "session_id": session_id
                })

            if is_correct and can_score:
                if user_id not in correct_players:
                    correct_players[user_id] = {'name': user_name}
                    if first_correct_answer_time == 0:
                        first_correct_answer_time = time.time()
                        end_game_early = self.config.get("end_game_after_bonus_time", True)
                        if end_game_early and bonus_time > 0:
                            asyncio.create_task(
                                asyncio.sleep(bonus_time),
                                name=f"end_game_task_{session_id}"
                            ).add_done_callback(
                                lambda _: not game_ended_by_attempts and controller.stop()
                            )

            if max_guess_attempts > 0 and guess_attempts_count >= max_guess_attempts:
                game_ended_by_attempts = True
                controller.stop()

        try:
            await unified_waiter(event)
        except TimeoutError:
            pass
        finally:
            self.last_game_end_time[session_id] = time.time()
            if session_id in self.context.active_game_sessions:
                self.context.active_game_sessions.remove(session_id)
        
        # 新增：在一轮游戏结束后，异步发送所有玩家的游戏数据
        if game_results_to_log:
            for result in game_results_to_log:
                asyncio.create_task(self._api_log_game(result))
        
        summary_prefix = f"本轮猜测已达上限({max_guess_attempts}次)！" if game_ended_by_attempts else "时间到！"
        if correct_players:
            winner_names = "、".join(player['name'] for player in correct_players.values())
            summary_text = f"{summary_prefix}\n本轮答对的玩家有：\n{winner_names}"
            await event.send(event.plain_result(summary_text))
        else:
            summary_text = f"{summary_prefix} 好像......没有人答对......"
            await event.send(event.plain_result(summary_text))
            
        await event.send(event.chain_result(answer_reveal_messages))


    @filter.command(
        "猜歌",
        alias={
            "gs",
            "猜歌1", "猜歌2", "猜歌3", "猜歌4", "猜歌5", "猜歌6", "猜歌7",
            "gs1", "gs2", "gs3", "gs4", "gs5", "gs6", "gs7"
        }
    )
    async def start_guess_song_unified(self, event: AstrMessageEvent):
        """统一处理所有固定模式的猜歌指令"""
        session_id = event.unified_msg_origin
        lock = self.context.game_session_locks.setdefault(session_id, asyncio.Lock())
        
        # 1. 解析模式
        match = re.search(r'(\d+)', event.message_str)
        mode_key = match.group(1) if match else 'normal'

        # 2. 轻量模式处理
        if self.lightweight_mode and mode_key in ['1', '2']:
            original_mode_name = self.game_modes[mode_key]['name']
            await event.send(event.plain_result(f'......轻量模式已启用，模式"{original_mode_name}"已自动切换为普通模式。'))
            mode_key = 'normal'
        
        # 3. 锁定并检查条件
        async with lock:
            can_start, message = await self._check_game_start_conditions(event)
            if not can_start:
                if message:
                    await event.send(event.plain_result(message))
                return
            self.context.active_game_sessions.add(session_id)

        # 4. 游戏确认开始，消耗次数
        initiator_id = event.get_sender_id()
        initiator_name = event.get_sender_name()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self.executor, self._consume_daily_play_attempt_sync, initiator_id, initiator_name
        )

        # 5. 获取配置并开始游戏
        mode_config = self.game_modes.get(mode_key)
        if not mode_config:
            await event.send(event.plain_result(f"......未知的猜歌模式 '{mode_key}'。"))
            if session_id in self.context.active_game_sessions:
                self.context.active_game_sessions.remove(session_id)
            return
            
        game_kwargs = mode_config['kwargs'].copy()
        game_kwargs['score'] = mode_config.get('score', 1) # 新增此行，将配置中的分数传递下去

        # 优化：优先使用预处理模式名作为game_type，否则使用mode_key
        if 'play_preprocessed' in game_kwargs:
            game_type_suffix = game_kwargs['play_preprocessed']
        elif 'melody_to_piano' in game_kwargs:
            game_type_suffix = 'piano'
        elif 'reverse_audio' in game_kwargs:
            game_type_suffix = 'reverse'
        elif 'speed_multiplier' in game_kwargs:
            game_type_suffix = 'speed_2x'
        else:
            game_type_suffix = 'normal'
            
        game_kwargs['game_type'] = f"guess_song_{game_type_suffix}"


        self._record_game_start(event.get_sender_id(), event.get_sender_name())
        asyncio.create_task(self._api_ping("guess_song"))
        
        try:
            game_data = await self.start_new_game(**game_kwargs)
        except Exception as e:
            logger.error(f"在 start_new_game 中发生错误: {e}", exc_info=True)
            game_data = None

        if not game_data:
            await event.send(event.plain_result("......开始游戏失败，可能是缺少资源文件或配置错误。"))
            if session_id in self.context.active_game_sessions:
                self.context.active_game_sessions.remove(session_id)
            return

        if not self.available_songs:
            await event.send(event.plain_result("......歌曲数据未加载，无法生成选项。"))
            if session_id in self.context.active_game_sessions:
                self.context.active_game_sessions.remove(session_id)
            return

        correct_song = game_data['song']
        other_songs = random.sample([s for s in self.available_songs if s['id'] != correct_song['id']], 11)
        options = [correct_song] + other_songs
        random.shuffle(options)
        
        game_data['options'] = options
        game_data['correct_answer_num'] = options.index(correct_song) + 1
        
        logger.info(f"[猜歌插件] 新游戏开始. 答案: {correct_song['title']} (选项 {game_data['correct_answer_num']})")
        
        options_img_path = await self._create_options_image(options)
        timeout_seconds = self.config.get("answer_timeout", 30)
        intro_text = f".......嗯\n这首歌是？请在{timeout_seconds}秒内发送编号回答。\n"
        
        intro_messages = [Comp.Plain(intro_text)]
        if options_img_path:
            intro_messages.append(Comp.Image(file=options_img_path))
        
        jacket_source = self._get_resource_path_or_url(f"music_jacket/{correct_song['jacketAssetbundleName']}.png")
        answer_reveal_messages = [
            Comp.Plain(f"正确答案是: {game_data['correct_answer_num']}. {correct_song['title']}\n"),
        ]
        if jacket_source:
            answer_reveal_messages.append(Comp.Image(file=str(jacket_source)))
        
        await self._run_game_session(event, game_data, intro_messages, answer_reveal_messages)

    @filter.command("随机猜歌", alias={"rgs"})
    async def start_random_guess_song(self, event: AstrMessageEvent):
        """开始一轮随机特殊模式的猜歌，采用目标优先模型选择效果"""
        session_id = event.unified_msg_origin
        lock = self.context.game_session_locks.setdefault(session_id, asyncio.Lock())
        
        async with lock:
            can_start, message = await self._check_game_start_conditions(event)
            if not can_start:
                if message:
                    await event.send(event.plain_result(message))
                return
            self.context.active_game_sessions.add(session_id)

        initiator_id = event.get_sender_id()
        initiator_name = event.get_sender_name()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self.executor, self._consume_daily_play_attempt_sync, initiator_id, initiator_name
        )
        
        self._record_game_start(event.get_sender_id(), event.get_sender_name())
        asyncio.create_task(self._api_ping("guess_song_random"))


        # --- 新的随机效果选择逻辑 (目标优先模型) ---

        # 1. 预计算所有可行的效果组合，并按分数分组
        all_combinations_by_score = self._precompute_random_combinations()

        if not all_combinations_by_score:
            await event.send(event.plain_result("......随机模式启动失败，没有可用的效果组合。请检查资源文件。"))
            if session_id in self.context.active_game_sessions:
                self.context.active_game_sessions.remove(session_id)
            return

        # 2. 生成目标分数概率分布
        target_distribution = self._get_random_target_distribution(all_combinations_by_score)

        # 3. 根据概率分布，随机选择一个目标分数
        scores = list(target_distribution.keys())
        probabilities = list(target_distribution.values())
        target_score = random.choices(scores, weights=probabilities, k=1)[0]
        
        # 4. 从能达成该分数的所有组合中，随机选择一个（已预处理好的）
        valid_combinations = all_combinations_by_score[target_score]
        chosen_processed_combo = random.choice(valid_combinations)

        # --- 从选中的组合中提取最终参数 ---
        combined_kwargs = chosen_processed_combo['final_kwargs']
        total_score = chosen_processed_combo['final_score']
        combined_kwargs['score'] = total_score
        combined_kwargs['game_type'] = 'guess_song_random'
        
        effect_names = [eff['name'] for eff in chosen_processed_combo['effects_list']]
        
        # 修正组合效果的额外加分显示
        effect_names_display = sorted(list(set(effect_names)))
        speed_mult = combined_kwargs.get('speed_multiplier')
        has_reverse = 'reverse_audio' in combined_kwargs

        if speed_mult and has_reverse:
            # 移除单独的名字，换成组合名
            effect_names_display = [n for n in effect_names_display if n not in ['倒放', '2倍速', '1.5倍速']]
            effect_names_display.append(f"倒放+{speed_mult}倍速组合(+1分)")
        
        # 记录本轮随机题型组合名 (用于统计)
        mode_name_str = '+'.join(sorted([name.replace(' ver.', '') for name in effect_names if name != 'Off']))
        combined_kwargs['random_mode_name'] = f"random_{mode_name_str}"
        
        # 显示将要应用的效果
        effects_text = "、".join(sorted(effect_names_display))
        await event.send(event.plain_result(f"......本轮应用效果：【{effects_text}】(总计{total_score}分)"))
        
        # --- 后续游戏逻辑 (与原版保持一致) ---
        try:
            game_data = await self.start_new_game(**combined_kwargs)
        except Exception as e:
            logger.error(f"在 start_new_game 中发生错误: {e}", exc_info=True)
            game_data = None

        if not game_data:
            error_message = game_data.get("error") if isinstance(game_data, dict) else "开始游戏失败，可能是缺少资源文件或配置错误。"
            await event.send(event.plain_result(f"......{error_message}"))
            if session_id in self.context.active_game_sessions:
                self.context.active_game_sessions.remove(session_id)
            return

        if not self.available_songs:
            await event.send(event.plain_result("......歌曲数据未加载，无法生成选项。"))
            if session_id in self.context.active_game_sessions:
                self.context.active_game_sessions.remove(session_id)
            return

        correct_song = game_data['song']
        other_songs = random.sample([s for s in self.available_songs if s['id'] != correct_song['id']], 11)
        options = [correct_song] + other_songs
        random.shuffle(options)
        
        game_data['options'] = options
        game_data['correct_answer_num'] = options.index(correct_song) + 1
        
        logger.info(f"[猜歌插件] 新游戏开始. 答案: {correct_song['title']} (选项 {game_data['correct_answer_num']})")
        
        options_img_path = await self._create_options_image(options)
        timeout_seconds = self.config.get("answer_timeout", 30)
        intro_text = f".......嗯\n这首歌是？请在{timeout_seconds}秒内发送编号回答。\n"
        
        intro_messages = [Comp.Plain(intro_text)]
        if options_img_path:
            intro_messages.append(Comp.Image(file=options_img_path))
        
        jacket_source = self._get_resource_path_or_url(f"music_jacket/{correct_song['jacketAssetbundleName']}.png")
        answer_reveal_messages = [
            Comp.Plain(f"正确答案是: {game_data['correct_answer_num']}. {correct_song['title']}\n"),
        ]
        if jacket_source:
            answer_reveal_messages.append(Comp.Image(file=str(jacket_source)))
        
        await self._run_game_session(event, game_data, intro_messages, answer_reveal_messages)

    @filter.command("猜歌手")
    async def start_vocalist_game(self, event: AstrMessageEvent):
        """开始一轮 '猜歌手' 游戏"""
        if not self.another_vocal_songs:
            await event.send(event.plain_result("......抱歉，没有找到包含 another_vocal 的歌曲，无法开始游戏。"))
            return

        session_id = event.unified_msg_origin
        lock = self.context.game_session_locks.setdefault(session_id, asyncio.Lock())
        
        async with lock:
            # 1. 先检查条件
            can_start, message = await self._check_game_start_conditions(event)
            if not can_start:
                if message:
                    await event.send(event.plain_result(message))
                return
            # 2. 检查通过后，立即在锁内标记会话状态
            self.context.active_game_sessions.add(session_id)

        # 3. 确认游戏开始后，在锁外为发起者扣减次数 (这是移动后的新位置)
        initiator_id = event.get_sender_id()
        initiator_name = event.get_sender_name()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self.executor, self._consume_daily_play_attempt_sync, initiator_id, initiator_name
        )
        
        debug_mode = self.config.get("debug_mode", False)

        # --- 新增：为猜歌手模式发送统计信标 ---
        asyncio.create_task(self._api_ping("guess_song_vocalist"))
        if not debug_mode:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self.executor, self._record_game_start, event.get_sender_id(), event.get_sender_name()
            )
        
        # 1. 准备游戏数据
        song = random.choice(self.another_vocal_songs)
        
        all_vocals = song.get('vocals', [])
        another_vocals = [v for v in all_vocals if v.get('musicVocalType') == 'another_vocal']
        
        if not another_vocals:
            await event.send(event.plain_result("......没有找到合适的歌曲版本，游戏无法开始。"))
            if session_id in self.context.active_game_sessions:
                self.context.active_game_sessions.remove(session_id)
            return
            
        correct_vocal_version = random.choice(another_vocals)
        
        try:
            game_data = await self.start_new_game(
                force_song_object=song,
                force_vocal_version=correct_vocal_version,
                speed_multiplier=1.5,
                game_type='guess_song_vocalist',
                guess_type='vocalist',
                mode_name='猜歌手'
            )
        except Exception as e:
            logger.error(f"在 start_new_game (猜歌手) 中发生错误: {e}", exc_info=True)
            game_data = None

        if not game_data:
            await event.send(event.plain_result("......准备音频失败，游戏无法开始。"))
            if session_id in self.context.active_game_sessions:
                self.context.active_game_sessions.remove(session_id)
            return

        # 2. 生成选项
        random.shuffle(another_vocals)
        game_data['num_options'] = len(another_vocals)
        game_data['correct_answer_num'] = another_vocals.index(correct_vocal_version) + 1
        game_data['game_mode'] = 'vocalist' # 标记为猜歌手模式，用于答案判断

        def get_vocalist_name(vocal_info):
            """[内部函数] 从vocal信息中获取格式化的歌手名。"""
            char_list = vocal_info.get('characters', [])
            if not char_list:
                return "未知"
            
            char_names = []
            for char in char_list:
                char_id = char.get('characterId')
                char_data = self.character_data.get(str(char_id))
                if char_data:
                    char_names.append(char_data.get("fullName", char_data.get("name", "未知")))
                else:
                    char_names.append("未知")
            return ' + '.join(char_names)

        
        compact_options_text = ""
        for i, vocal in enumerate(another_vocals):
            vocalist_name = get_vocalist_name(vocal)
            compact_options_text += f"{i + 1}. {vocalist_name}\n"
        
        # 3. 准备消息
        timeout_seconds = self.config.get("answer_timeout", 30)
        intro_text = f"这首歌是【{song['title']}】，正在演唱的是谁？[1.5倍速]\n请在{timeout_seconds}秒内发送编号回答。\n\n⚠️ 测试功能，不计分\n\n{compact_options_text}"
        jacket_source = self._get_resource_path_or_url(f"music_jacket/{song['jacketAssetbundleName']}.png")
        
        intro_messages = [Comp.Plain(intro_text)]
        if jacket_source:
            intro_messages.append(Comp.Image(file=str(jacket_source)))
            
        correct_vocalist_name = get_vocalist_name(correct_vocal_version)
        answer_reveal_messages = [
            Comp.Plain(f"正确答案是: {game_data['correct_answer_num']}. {correct_vocalist_name}")
        ]

        # 4. 启动游戏
        await self._run_game_session(event, game_data, intro_messages, answer_reveal_messages)

    @filter.command("猜歌帮助")
    async def show_guess_song_help(self, event: AstrMessageEvent):
        """以图片形式显示猜歌插件帮助。"""
        if not self._is_group_allowed(event):
            return

        loop = asyncio.get_running_loop()
        try:
            img_path = await loop.run_in_executor(self.executor, self._draw_help_image_sync)
            if img_path:
                await event.send(event.image_result(img_path))
            else:
                await event.send(event.plain_result("生成帮助图片时出错。"))
        except Exception as e:
            logger.error(f"发送帮助图片时发生错误: {e}", exc_info=True)
            await event.send(event.plain_result("生成帮助图片时出错，请查看日志。"))

    @filter.command("群猜歌排行榜", alias={"gssrank", "gstop"})
    async def show_ranking(self, event: AstrMessageEvent):
        """显示当前群聊的猜歌排行榜"""
        if not self._is_group_allowed(event): return

        session_id = event.unified_msg_origin
        loop = asyncio.get_running_loop()
        rows = await loop.run_in_executor(self.executor, self._get_ranking_data_sync, session_id)

        if not rows:
            await event.send(event.plain_result("......本群目前还没有人参与过猜歌游戏"))
            return

        img_path = await loop.run_in_executor(
            self.executor, self._draw_ranking_image_sync, rows[:10], "本群猜歌排行榜"
        )
        if img_path:
            await event.send(event.image_result(img_path))
        else:
            await event.send(event.plain_result("生成排行榜图片时出错。"))

    def _get_ranking_data_sync(self, session_id: str):
        """获取指定会话的排行榜数据 (前10名)，现在由核心数据源驱动。"""
        full_ranking = self._get_full_group_ranking_sync(session_id)
        # 为了兼容绘图函数，即使完整的排行榜不足10人，也返回所有数据
        return full_ranking[:10]

    def _draw_ranking_image_sync(self, rows, title_text="猜歌排行榜") -> Optional[str]:
        """[辅助函数] 同步绘制排行榜图片，已适配服务器/本群两种模式"""
        try:
            # 移植猜卡插件的排行榜生成逻辑以获得更好看的样式
            width, height = 650, 950
            bg_color_start, bg_color_end = (230, 240, 255), (200, 210, 240)
            img = Image.new("RGB", (width, height), bg_color_start)
            draw_bg = ImageDraw.Draw(img)
            for y in range(height):
                r = int(bg_color_start[0] + (bg_color_end[0] - bg_color_start[0]) * y / height)
                g = int(bg_color_start[1] + (bg_color_end[1] - bg_color_start[1]) * y / height)
                b = int(bg_color_start[2] + (bg_color_end[2] - bg_color_start[2]) * y / height)
                draw_bg.line([(0, y), (width, y)], fill=(r, g, b))

            background_path = self.resources_dir / "ranking_bg.png"
            if background_path.exists():
                try:
                    custom_bg = Image.open(background_path).convert("RGBA").resize((width, height), LANCZOS)
                    custom_bg.putalpha(128)
                    img = img.convert("RGBA")
                    img = Image.alpha_composite(img, custom_bg)
                except Exception as e:
                    logger.warning(f"加载或混合自定义背景图片失败: {e}")

            if img.mode != 'RGBA': img = img.convert('RGBA')
            white_overlay = Image.new("RGBA", img.size, (255, 255, 255, 100))
            img = Image.alpha_composite(img, white_overlay)

            font_color, shadow_color = (30, 30, 50), (180, 180, 190, 128)
            header_color, score_color, accuracy_color = (80, 90, 120), (235, 120, 20), (0, 128, 128)

            try:
                font_path = self.resources_dir / "font.ttf"
                title_font = ImageFont.truetype(str(font_path), 48)
                header_font = ImageFont.truetype(str(font_path), 28)
                body_font = ImageFont.truetype(str(font_path), 26)
                id_font = ImageFont.truetype(str(font_path), 16)
                medal_font = ImageFont.truetype(str(font_path), 36)
            except IOError:
                title_font, header_font, body_font, id_font = [ImageFont.load_default()] * 4
                medal_font = body_font

            with Pilmoji(img) as pilmoji:
                # 标题文本 = "猜歌排行榜"
                # 修正：将浮点数坐标转换为整数
                center_x, title_y = int(width / 2), 80
                pilmoji.text((center_x + 2, title_y + 2), title_text, font=title_font, fill=shadow_color, anchor="mm")
                pilmoji.text((center_x, title_y), title_text, font=title_font, fill=font_color, anchor="mm")

                headers = ["排名", "玩家", "总分", "正确率", "总次数"]
                col_positions_header = [40, 120, 320, 450, 560]
                current_y = title_y + int(pilmoji.getsize(title_text, font=title_font)[1] / 2) + 45
                for i, header in enumerate(headers):
                    pilmoji.text((col_positions_header[i], current_y), header, font=header_font, fill=header_color)

                current_y += 55
                rank_icons = ["🥇", "🥈", "🥉"]
                
                for i, row in enumerate(rows):
                    user_id, user_name, score, attempts, correct_attempts = str(row[0]), row[1], str(row[2]), str(row[3]), row[4]
                    
                    if attempts == -1:
                        accuracy = "N/A"
                        attempts_str = "N/A"
                    else:
                        attempts_str = str(attempts)
                        accuracy = f"{(correct_attempts * 100 / int(attempts) if int(attempts) > 0 else 0):.1f}%"
                    
                    rank = i + 1
                    col_positions = [40, 120, 320, 450, 560]
                    pilmoji.text((100, current_y), str(rank), font=body_font, fill=font_color, anchor="ra")
                    if i < 3:
                        pilmoji.text((col_positions[0], current_y - 2), rank_icons[i], font=medal_font, fill=font_color)

                    max_name_width = col_positions[2] - col_positions[1] - 20
                    if body_font.getbbox(user_name)[2] > max_name_width:
                        while body_font.getbbox(user_name + "...")[2] > max_name_width and len(user_name) > 0:
                            user_name = user_name[:-1]
                        user_name += "..."

                    pilmoji.text((col_positions[1], current_y), user_name, font=body_font, fill=font_color)
                    pilmoji.text((col_positions[1], current_y + 32), f"ID: {user_id}", font=id_font, fill=header_color)
                    pilmoji.text((col_positions[2], current_y), score, font=body_font, fill=score_color)
                    pilmoji.text((col_positions[3], current_y), accuracy, font=body_font, fill=accuracy_color)
                    pilmoji.text((col_positions[4], current_y), attempts_str, font=body_font, fill=font_color)

                    if i < len(rows) - 1:
                        draw = ImageDraw.Draw(img)
                        draw.line([(30, current_y + 60), (width - 30, current_y + 60)], fill=(200, 200, 210, 128), width=1)
                    
                    current_y += 70

                footer_text = f"GuessSong v{PLUGIN_VERSION} | Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                pilmoji.text((center_x, height - 25), footer_text, font=id_font, fill=header_color, anchor="ms")

            img_path = self.output_dir / f"song_ranking_{int(time.time())}.png"
            img.save(img_path)
            return str(img_path)

        except Exception as e:
            logger.error(f"生成猜歌排行榜图片时出错: {e}", exc_info=True)
            return None

    def _get_global_ranking_data_sync(self):
        """获取全局排行榜数据。"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_id, user_name, SUM(score) as total_score, SUM(attempts) as total_attempts, SUM(correct_attempts) as total_correct
                FROM user_stats
                GROUP BY user_id, user_name
                ORDER BY total_score DESC
                LIMIT 10
            """)
            return cursor.fetchall()

    @filter.command("本地猜歌排行榜", alias={"localrank"})
    async def show_local_global_ranking(self, event: AstrMessageEvent):
        """显示本地存储的全局猜歌排行榜"""
        if not self._is_group_allowed(event): return
        
        loop = asyncio.get_running_loop()
        rows = await loop.run_in_executor(self.executor, self._get_global_ranking_data_sync)

        if not rows:
            await event.send(event.plain_result("......目前还没有人参与过猜歌游戏"))
            return
            
        # 为绘图函数重新格式化行数据
        formatted_rows = [(row[0], row[1], row[2], row[3], row[4]) for row in rows]

        img_path = await loop.run_in_executor(
            self.executor, self._draw_ranking_image_sync, formatted_rows[:10], "本地总排行榜"
        )
        if img_path:
            await event.send(event.image_result(img_path))
        else:
            await event.send(event.plain_result("生成排行榜图片时出错。"))

    @filter.command("猜歌排行榜", alias={"gslrank", "gslglobal"})
    async def show_global_ranking(self, event: AstrMessageEvent):
        """显示服务器猜歌排行榜（已最终修正数据格式问题）"""
        if not self.api_key:
            yield event.plain_result("......未配置API Key，无法获取服务器排行榜。")
            return
        
        if not self.stats_server_url:
            yield event.plain_result("......服务器地址配置不正确。")
            return

        try:
            session = await self._get_session()
            if not session:
                yield event.plain_result("......网络组件初始化失败。")
                return

            leaderboard_url = f"{self.stats_server_url}/api/leaderboard"
            
            async with session.get(leaderboard_url, headers=self._get_api_headers(), timeout=10) as response:
                if response.status == 401:
                    yield event.plain_result("......API密钥无效，无法获取服务器排行榜。请检查插件配置。")
                    return
                response.raise_for_status()
                rows_json = await response.json()

        except Exception as e:
            logger.error(f"获取服务器排行榜失败: {e}", exc_info=True)
            yield event.plain_result(f"......获取服务器排行榜失败，请检查服务器状态和网络连接。错误: {e}")
            return

        if not rows_json:
            yield event.plain_result("......服务器排行榜上还没有任何数据。")
            return

        # 为绘图函数重新格式化行数据
        formatted_rows = [
            (
                r.get('user_id'),
                r.get('user_name'),
                r.get('total_score', 0),
                r.get('total_attempts', 0),
                r.get('correct_attempts', 0)
            )
            for r in rows_json
        ]

        try:
            loop = asyncio.get_running_loop()
            img_path = await loop.run_in_executor(self.executor, self._draw_ranking_image_sync, formatted_rows[:10], "服务器猜歌排行榜")
            if img_path:
                yield event.image_result(img_path)
            else:
                yield event.plain_result("生成排行榜图片时出错。")
        except Exception as e:
            logger.error(f"绘制服务器排行榜图片时出错: {e}", exc_info=True)
            yield event.plain_result("生成排行榜图片时出错。")


    @filter.command("猜歌分数", alias={"gsscore", "我的猜歌分数"})
    async def show_user_score(self, event: AstrMessageEvent):
        """显示用户在本群、服务器和本地的总分数统计。"""
        user_id = str(event.get_sender_id())
        user_name = event.get_sender_name()
        # 修正：与 /群猜歌排行榜 统一使用 event.unified_msg_origin 作为群聊的唯一标识
        session_id = event.unified_msg_origin
        
        # 使用 asyncio.gather 并发执行所有异步和同步（在线程池中）任务
        server_stats_task = asyncio.create_task(self._api_get_user_global_stats(user_id))
        
        loop = asyncio.get_running_loop()
        # 统一调用新的数据获取函数
        group_stats_task = loop.run_in_executor(self.executor, self._get_user_stats_in_group_sync, user_id, session_id)
        local_global_stats_task = loop.run_in_executor(self.executor, self._get_user_local_global_stats_sync, user_id)
        
        server_stats, group_stats, local_global_stats = await asyncio.gather(
            server_stats_task, group_stats_task, local_global_stats_task
        )
        
        # 构建最终输出
        result_parts = [f"📊 {user_name} 的猜歌报告"]
        
        # 1. 本群战绩
        if group_stats:
            group_score = group_stats.get('score', 0)
            group_attempts = group_stats.get('attempts', -1)
            group_correct = group_stats.get('correct_attempts', -1)
            
            rank_str = f"(排名: {group_stats['rank']})" if group_stats.get('rank') is not None else "(排名: N/A)"
            
            # 只有在新数据模型（attempts >= 0）下才显示正确率
            if group_attempts >= 0:
                accuracy_str = f"{(group_correct * 100 / group_attempts if group_attempts > 0 else 0):.1f}% ({group_correct}/{group_attempts})"
            else:
                accuracy_str = "N/A" # 旧数据无法计算正确率

            result_parts.append(
                f"⚜️ 本群战绩 {rank_str}\n"
                f"   - 分数: {group_score}\n"
                f"   - 正确率: {accuracy_str}"
            )
        else:
            result_parts.append(
                "⚜️ 本群战绩\n"
                "   - 暂无记录"
            )
        # 2. 总计战绩（服务器同步）
        if server_stats:
            server_score = server_stats.get('total_score', 0)
            server_rank = server_stats.get('rank', 'N/A')
            server_attempts = server_stats.get('total_attempts', 0)
            server_correct = server_stats.get('correct_attempts', 0)
            accuracy = f"{(server_correct * 100 / server_attempts if server_attempts > 0 else 0):.1f}%"
            
            result_parts.append(
                f"🌐 总计战绩 (服务器, 排名: {server_rank})\n"
                f"   - 分数: {server_score}\n"
                f"   - 正确率: {accuracy} ({server_correct}/{server_attempts})"
            )
        # 如果服务器数据不可用，则显示本地的全局数据作为备用
        elif local_global_stats:
            local_score = local_global_stats.get('score', 0)
            local_rank = local_global_stats.get('rank', 'N/A')
            local_attempts = local_global_stats.get('attempts', 0)
            local_correct = local_global_stats.get('correct', 0)
            accuracy = f"{(local_correct * 100 / local_attempts if local_attempts > 0 else 0):.1f}%"
            
            result_parts.append(
                f"🌐 总计战绩 (仅本地, 排名: {local_rank})\n"
                f"   - 分数: {local_score}\n"
                f"   - 正确率: {accuracy} ({local_correct}/{local_attempts})"
            )
        else:
             result_parts.append(
                "🌐 总计战绩\n"
                "   - 暂无记录"
            )

        # 3. 每日剩余次数
        if local_global_stats:
            today = datetime.now().strftime("%Y-%m-%d")
            # 只有当最后游戏日期是今天时，才显示已玩次数
            daily_plays = local_global_stats.get('daily_plays', 0)
            last_play_date = local_global_stats.get('last_play_date', '')
            games_today = daily_plays if last_play_date == today else 0
            
            play_limit = self.config.get("daily_play_limit", 15)
            listen_limit = self.config.get("daily_listen_limit", 5)
            
            # 同样的方法获取听歌次数
            can_listen, listen_today = self._get_user_daily_limits_sync(user_id)
            
            result_parts.append(
                f"🕒 剩余次数\n"
                f"   - 猜歌: {play_limit - games_today}/{play_limit}\n"
                f"   - 听歌: {listen_limit - listen_today}/{listen_limit}"
            )

        await event.send(event.plain_result("\n\n".join(result_parts)))

    def _get_user_daily_limits_sync(self, user_id: str) -> Tuple[bool, int]:
        """[辅助函数] 同步获取用户每日听歌限制。返回 (是否可听, 今天听歌次数)"""
        listen_limit = self.config.get("daily_listen_limit", 5)
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT daily_listen_songs, last_listen_date FROM user_stats WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                daily_listen, last_date = row
                today = datetime.now().strftime("%Y-%m-%d")
                if last_date != today:
                    return True, 0
                return daily_listen < listen_limit, daily_listen
            return True, 0
    
    # --- 数据和状态管理方法 ---
    def _record_game_start(self, user_id: str, user_name: str):
        # 此方法现在是占位符。逻辑在 _update_stats 和 _run_game_session 中处理。
        # 保留它是为了将来潜在的使用或兼容性。
        pass

    def _record_listen_song(self, user_id: str, user_name: str, session_id: str):
        """
        [已重构] 记录用户听歌次数，并同步处理每日状态重置，修复日期不一致的漏洞。
        """
        with self.get_conn() as conn:
            cursor = conn.cursor()
            today = datetime.now().strftime("%Y-%m-%d")

            # 1. 获取完整的用户每日统计数据
            cursor.execute("""
                SELECT daily_listen_songs, last_listen_date, daily_games_played, last_played_date
                FROM user_stats WHERE user_id = ?
            """, (user_id,))
            row = cursor.fetchone()

            if row:
                # 用户存在
                daily_listen, last_listen, daily_games, last_played = row
                
                # 2. 检查是否是新的一天，如果是，则重置所有每日统计
                if last_listen != today:
                    daily_listen = 0
                    # 关键修复：当听歌是当天第一项活动时，也重置游戏次数
                    if last_played != today:
                        daily_games = 0
                
                # 3. 更新听歌次数
                daily_listen += 1

                # 4. 将所有更新写回数据库，确保两个日期同步
                cursor.execute("""
                    UPDATE user_stats SET 
                        daily_listen_songs = ?, 
                        last_listen_date = ?, 
                        daily_games_played = ?, 
                        last_played_date = ?, 
                        user_name = ?
                    WHERE user_id = ?
                """, (daily_listen, today, daily_games, today, user_name, user_id))
            else:
                # 新用户，直接插入，确保两个日期都是今天
                cursor.execute("""
                    INSERT INTO user_stats (user_id, user_name, daily_listen_songs, last_listen_date, last_played_date) 
                    VALUES (?, ?, 1, ?, ?)
                """, (user_id, user_name, today, today))
            
            conn.commit()

    def _update_stats(self, session_id: str, user_id: str, user_name: str, score: int, correct: bool):
        """
        [核心] 同步更新用户统计数据。
        - 确保用户存在于数据库中。
        - 更新总分、总尝试次数、总正确次数、连胜纪录。
        - 更新分群JSON分数。
        - 更新每日游戏次数和最后游戏日期。
        """
        with self.get_conn() as conn:
            cursor = conn.cursor()
            today = datetime.now().strftime("%Y-%m-%d")

            # 1. 检查用户是否存在，如果不存在则创建
            cursor.execute("SELECT * FROM user_stats WHERE user_id = ?", (user_id,))
            user_data = cursor.fetchone()

            if user_data is None:
                # 插入一个包含默认值的新用户记录
                cursor.execute("""
                    INSERT INTO user_stats (
                        user_id, user_name, score, attempts, correct_attempts, 
                        daily_games_played, last_played_date, daily_listen_songs, 
                        last_listen_date, correct_streak, max_correct_streak, group_scores
                    ) VALUES (?, ?, 0, 0, 0, 0, ?, 0, ?, 0, 0, '{}')
                """, (user_id, user_name, today, today))
                # 重新获取新创建的用户数据
                cursor.execute("SELECT * FROM user_stats WHERE user_id = ?", (user_id,))
                user_data = cursor.fetchone()

            # 为了方便通过列名访问，将元组转换为字典
            columns = [desc[0] for desc in cursor.description]
            user_stats = dict(zip(columns, user_data))
            
      

            # 3. 更新核心统计数据
            user_stats['attempts'] += 1
            if correct:
                user_stats['correct_attempts'] += 1
                user_stats['score'] += score
                user_stats['correct_streak'] += 1
                user_stats['max_correct_streak'] = max(user_stats['correct_streak'], user_stats['max_correct_streak'])
            else:
                user_stats['correct_streak'] = 0

            # 4. 更新分群分数 (group_scores JSON) -> 已升级为记录详细统计
            try:
                group_scores_json = user_stats.get('group_scores', '{}')
                group_scores = json.loads(group_scores_json or '{}')
            except (json.JSONDecodeError, TypeError):
                group_scores = {}

            # 获取当前群组的统计数据
            group_stat_raw = group_scores.get(session_id)

            # [迁移逻辑] 处理从旧的纯分数格式到新的字典格式的转换
            if isinstance(group_stat_raw, int):
                # 这是旧格式，需要转换为新格式
                # 我们无法追溯历史尝试次数，所以只能从现在开始计算
                group_stat = {"score": group_stat_raw, "attempts": 0, "correct_attempts": 0}
            elif isinstance(group_stat_raw, dict):
                # 已经是新格式
                group_stat = group_stat_raw
            else:
                # 不存在或格式错误，初始化
                group_stat = {"score": 0, "attempts": 0, "correct_attempts": 0}
            
            # 更新群组统计
            group_stat["score"] += score
            group_stat["attempts"] += 1
            if correct:
                group_stat["correct_attempts"] += 1
            
            # 将更新后的群组统计写回
            group_scores[session_id] = group_stat
            updated_group_scores_json = json.dumps(group_scores)

            # 5. 将所有更新写回数据库
            cursor.execute("""
                UPDATE user_stats SET
                    user_name = ?,
                    score = ?,
                    attempts = ?,
                    correct_attempts = ?,
                    correct_streak = ?,
                    max_correct_streak = ?,
                    group_scores = ?
                WHERE user_id = ?
            """, (
                user_name,
                user_stats['score'],
                user_stats['attempts'],
                user_stats['correct_attempts'],
                user_stats['correct_streak'],
                user_stats['max_correct_streak'],
                updated_group_scores_json,
                user_id
            ))
            conn.commit()
    def _consume_daily_play_attempt_sync(self, user_id: str, user_name: str):
        """
        [新] 为指定用户消费一次每日游戏次数。
        此函数应在游戏确认开始后，只对发起者调用一次。
        """
        with self.get_conn() as conn:
            cursor = conn.cursor()
            today = datetime.now().strftime("%Y-%m-%d")
            
            # 检查用户记录是否存在
            cursor.execute("SELECT daily_games_played, last_played_date FROM user_stats WHERE user_id = ?", (user_id,))
            user_data = cursor.fetchone()

            if user_data:
                # 用户存在，更新次数
                daily_games, last_played = user_data
                if last_played != today:
                    # 如果不是今天玩的，次数重置为1
                    new_daily_games = 1
                else:
                    # 如果是今天玩的，次数加1
                    new_daily_games = daily_games + 1
                
                cursor.execute(
                    "UPDATE user_stats SET daily_games_played = ?, last_played_date = ?, user_name = ? WHERE user_id = ?",
                    (new_daily_games, today, user_name, user_id)
                )
            else:
                # 用户不存在，创建新记录
                cursor.execute(
                    """
                    INSERT INTO user_stats (user_id, user_name, daily_games_played, last_played_date)
                    VALUES (?, ?, 1, ?)
                    """, (user_id, user_name, today)
                )
            conn.commit()

    def _can_play(self, user_id: str) -> bool:
        """检查用户今天是否还能玩游戏。"""
        daily_limit = self.config.get("daily_play_limit", 15)
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT daily_games_played, last_played_date FROM user_stats WHERE user_id = ?", (user_id,))
            user_data = cursor.fetchone()
            return not (user_data and user_data[1] == time.strftime("%Y-%m-%d") and user_data[0] >= daily_limit)
            

    @filter.command("重置猜歌次数", alias={"resetgs"})
    async def reset_guess_limit(self, event: AstrMessageEvent):
        """重置用户猜歌次数（仅限管理员）"""
        if not event.is_admin:
            return
            
        parts = event.message_str.strip().split()
        if len(parts) > 1 and parts[1].isdigit():
            target_id = parts[1]
            success = await asyncio.to_thread(self._reset_guess_limit_sync, target_id)
            if success:
                await event.send(event.plain_result(f"......用户 {target_id} 的猜歌次数已重置。"))
            else:
                await event.send(event.plain_result(f"......未找到用户 {target_id} 的游戏记录。"))
        else:
            await event.send(event.plain_result("请提供要重置的用户ID。"))

    def _reset_guess_limit_sync(self, target_id: str) -> bool:
        """同步重置指定用户的每日游戏次数。"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            # 这会重置该用户在所有会话中的每日计数。
            cursor.execute("UPDATE user_stats SET daily_games_played = 0 WHERE user_id = ?", (target_id,))
            conn.commit()
            return cursor.rowcount > 0

    @filter.command("重置听歌次数", alias={"resetls"})
    async def reset_listen_limit(self, event: AstrMessageEvent):
        """重置用户每日听歌次数（仅限管理员）"""
        if not event.is_admin:
            return
            
        parts = event.message_str.strip().split()
        if len(parts) > 1 and parts[1].isdigit():
            target_id = parts[1]
            success = await asyncio.to_thread(self._reset_listen_limit_sync, target_id)
            if success:
                await event.send(event.plain_result(f"......用户 {target_id} 的听歌次数已重置。"))
            else:
                await event.send(event.plain_result(f"......未找到用户 {target_id} 的游戏记录。"))
        else:
            await event.send(event.plain_result("请提供要重置的用户ID。"))

    def _reset_listen_limit_sync(self, target_id: str) -> bool:
        """同步重置指定用户的每日听歌次数。"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE user_stats SET daily_listen_songs = 0 WHERE user_id = ?", (target_id,))
            conn.commit()
            return cursor.rowcount > 0

    def _reset_mode_stats_sync(self):
        """同步清空题型统计数据。"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM mode_stats")
            conn.commit()

    @filter.command("重置题型统计", alias={"resetmodestats"})
    async def reset_mode_stats(self, event: AstrMessageEvent):
        """清空所有题型统计数据（仅限管理员）"""
        if str(event.get_sender_id()) not in self.config.get("super_users", []):
            return
        
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.executor, self._reset_mode_stats_sync)
        
        await event.send(event.plain_result("......所有题型统计数据已被清空。"))

    async def terminate(self):
        """在插件终止时关闭线程池和后台任务"""
        logger.info("正在关闭猜歌插件的后台任务...")
        if hasattr(self, '_cleanup_task') and self._cleanup_task:
            self._cleanup_task.cancel()
        if hasattr(self, '_manifest_load_task') and self._manifest_load_task:
            self._manifest_load_task.cancel()
        
        # 新增：安全地关闭 aiohttp session
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
            logger.info("aiohttp session 已关闭。")

        self.executor.shutdown(wait=False)
        logger.info("猜歌插件已终止。")

    def _update_mode_stats(self, mode: str, correct: bool):
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT total_attempts, correct_attempts FROM mode_stats WHERE mode = ?", (mode,))
            row = cursor.fetchone()
            if row:
                total, correct_count = row
                total += 1
                correct_count += 1 if correct else 0
                cursor.execute("UPDATE mode_stats SET total_attempts = ?, correct_attempts = ? WHERE mode = ?", (total, correct_count, mode))
            else:
                cursor.execute("INSERT INTO mode_stats (mode, total_attempts, correct_attempts) VALUES (?, ?, ?)", (mode, 1, 1 if correct else 0))
            conn.commit() 

    @filter.command("查看统计", alias={"mode_stats", "题型统计"})
    async def show_mode_stats(self, event: AstrMessageEvent):
        """显示各题型的正确率统计（图片排行）"""
        if not self._is_group_allowed(event): return
        
        loop = asyncio.get_running_loop()
        rows = None

        if not self.api_key:
            # --- 离线模式：从本地数据库获取 ---
            rows = await loop.run_in_executor(self.executor, self._get_mode_stats_sync)
        else:
            # --- 在线模式：从服务器API获取 ---
            try:
                session = await self._get_session()
                if not session:
                    yield event.plain_result("......网络组件初始化失败。"); return
                
                stats_url = f"{self.stats_server_url}/api/mode_stats"
                async with session.get(stats_url, headers=self._get_api_headers(), timeout=5) as response:
                    if response.status == 401:
                        yield event.plain_result("......API密钥无效，无法获取题型统计。"); return
                    response.raise_for_status()
                    rows_json = await response.json()
                    # 将json字典列表转换为数据库游标返回的元组列表，以适配后续代码
                    rows = [(r['mode'], r['total_attempts'], r['correct_attempts']) for r in rows_json]
            except Exception as e:
                logger.error(f"获取在线题型统计失败: {e}", exc_info=True)
                yield event.plain_result("......获取在线题型统计失败，将尝试使用本地数据。")
                # --- 在线模式失败，回退到本地 ---
                rows = await loop.run_in_executor(self.executor, self._get_mode_stats_sync)

        if not rows:
            yield event.plain_result("暂无题型统计数据。"); return
        # 计算正确率并排序
        stats = []
        for mode, total, correct in rows:
            acc = (correct * 100 / total) if total > 0 else 0
            stats.append((mode, total, correct, acc))
        stats.sort(key=lambda x: x[3])  # 按正确率升序
        
        # 在线程池中生成图片
        img_path = await loop.run_in_executor(self.executor, self._draw_mode_stats_image_sync, stats[:10])

        if img_path:
            yield event.image_result(img_path) 
        else:
            yield event.plain_result("生成统计图片时出错。")

    def _get_mode_stats_sync(self):
        """[辅助函数] 同步获取题型统计数据"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT mode, total_attempts, correct_attempts FROM mode_stats")
            return cursor.fetchall()

    def _draw_mode_stats_image_sync(self, stats) -> Optional[str]:
        """[辅助函数] 同步绘制题型统计图片"""
        try:
            # 固定排行榜分辨率
            width, height = 650, 950
            bg_color_start, bg_color_end = (230, 240, 255), (200, 210, 240)
            img = Image.new("RGB", (width, height), bg_color_start)
            draw_bg = ImageDraw.Draw(img)
            for y in range(height):
                r = int(bg_color_start[0] + (bg_color_end[0] - bg_color_start[0]) * y / height)
                g = int(bg_color_start[1] + (bg_color_end[1] - bg_color_start[1]) * y / height)
                b = int(bg_color_start[2] + (bg_color_end[2] - bg_color_start[2]) * y / height)
                draw_bg.line([(0, y), (width, y)], fill=(r, g, b))
            background_path = self.resources_dir / "ranking_bg.png"
            if background_path.exists():
                try:
                    custom_bg = Image.open(background_path).convert("RGBA").resize((width, height), LANCZOS)
                    custom_bg.putalpha(128)
                    img = img.convert("RGBA")
                    img = Image.alpha_composite(img, custom_bg)
                except Exception as e:
                    logger.warning(f"加载或混合自定义背景图片失败: {e}")
            if img.mode != 'RGBA': img = img.convert('RGBA')
            white_overlay = Image.new("RGBA", img.size, (255, 255, 255, 100))
            img = Image.alpha_composite(img, white_overlay)
            font_color, shadow_color = (30, 30, 50), (180, 180, 190, 128)
            header_color, score_color, accuracy_color = (80, 90, 120), (235, 120, 20), (0, 128, 128)
            try:
                font_path = self.resources_dir / "font.ttf"
                title_font = ImageFont.truetype(str(font_path), 44)
                header_font = ImageFont.truetype(str(font_path), 28)
                body_font = ImageFont.truetype(str(font_path), 26)
            except IOError:
                title_font = header_font = body_font = ImageFont.load_default()
            with Pilmoji(img) as pilmoji:
                title_text = "题型正确率排行"
                center_x, title_y = int(width / 2), 60
                pilmoji.text((center_x + 2, title_y + 2), title_text, font=title_font, fill=shadow_color, anchor="mm")
                pilmoji.text((center_x, title_y), title_text, font=title_font, fill=font_color, anchor="mm")
                headers = ["题型", "答对/总数", "正确率"]
                col_positions = [60, 320, 500]
                current_y = title_y + 50
                for i, header in enumerate(headers):
                    pilmoji.text((col_positions[i], current_y), header, font=header_font, fill=header_color)
                current_y += 45
                max_mode_width = col_positions[1] - col_positions[0] - 20
                for i, (mode, total, correct, acc) in enumerate(stats):
                    mode_disp = self._mode_display_name(mode)
                    # 自动换行处理
                    lines = []
                    temp = ""
                    for ch in mode_disp:
                        if pilmoji.getsize(temp + ch, font=body_font)[0] > max_mode_width and temp:
                            lines.append(temp)
                            temp = ch
                        else:
                            temp += ch
                    if temp:
                        lines.append(temp)

                    # 动态计算行高和垂直居中位置
                    line_spacing = 32
                    block_height = line_spacing * (len(lines))
                    row_center_y = current_y + block_height / 2

                    for idx, line in enumerate(lines):
                        pilmoji.text((col_positions[0], current_y + idx * line_spacing + 5), line, font=body_font, fill=font_color)
                    
                    # 将统计数据与文本块垂直居中
                    pilmoji.text((col_positions[1], row_center_y), f"{correct}/{total}", font=body_font, fill=score_color, anchor="lm")
                    pilmoji.text((col_positions[2], row_center_y), f"{acc:.1f}%", font=body_font, fill=accuracy_color, anchor="lm")
                    
                    row_height = max(70, block_height + 15)
                    if i < len(stats) - 1:
                        draw = ImageDraw.Draw(img)
                        draw.line([(40, current_y + row_height - 8), (width - 40, current_y + row_height - 8)], fill=(200, 200, 210, 128), width=1)
                    current_y += row_height
                footer_text = f"GuessSong v{PLUGIN_VERSION} | Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                pilmoji.text((center_x, height - 25), footer_text, font=body_font, fill=header_color, anchor="ms")
            
            img_path = self.output_dir / f"mode_stats_{int(time.time())}.png"
            img.save(img_path)
            return str(img_path)
        except Exception as e:
            logger.error(f"生成题型统计图片时出错: {e}", exc_info=True)
            return None

    def _mode_display_name(self, mode_key: str) -> str:
        """(重构) 题型名美化，支持稳定ID"""
        default_map = {"normal": "普通"}
        if mode_key in default_map:
            return default_map[mode_key]

        if mode_key.startswith("random_"):
            # 解析稳定ID，例如 "random_bass+reverse"
            ids = mode_key.replace("random_", "").split('+')
            # 查找每个ID当前的显示名
            names = [self.game_effects.get(i, {}).get('name', i) for i in ids]
            return "随机-" + "+".join(names)
        
        # 兼容旧的简单模式名
        # 注意：这里我们假设简单模式的 mode_key 和 效果的 stable ID 是一致的
        return self.game_effects.get(mode_key, {}).get('name', mode_key)

    @filter.command("测试猜歌", alias={"test_song", "调试猜歌"})
    async def test_guess_song(self, event: AstrMessageEvent):
        """(管理员) 生成一个用于测试的猜歌游戏，可指定歌曲和多种模式。"""
        if str(event.get_sender_id()) not in self.config.get("super_users", []):
            return

        parts = event.message_str.strip().split(maxsplit=1)
        if len(parts) < 2:
            await event.send(event.plain_result("用法: /测试猜歌 [模式,...] <歌曲名或ID>\n例如: /测试猜歌 bass,reverse Tell Your World"))
            return

        args_str = parts[1]
        arg_parts = args_str.split()
        
        potential_modes_str = arg_parts[0]
        temp_modes = re.split(r'[,，]', potential_modes_str)
        
        parsed_mode_keys = []
        is_first_arg_modes = True
        for mode_str in temp_modes:
            mode_key = self.mode_name_map.get(mode_str.lower())
            if mode_key:
                parsed_mode_keys.append(mode_key)
            else:
                is_first_arg_modes = False
                break
        
        if is_first_arg_modes and parsed_mode_keys:
            mode_keys_input = list(dict.fromkeys(parsed_mode_keys))
            song_query = " ".join(arg_parts[1:])
        else:
            mode_keys_input = []
            song_query = args_str

        if not song_query:
            await event.send(event.plain_result("请输入要测试的歌曲名称或ID。"))
            return

        final_kwargs = {}
        effect_names = []
        total_score = 0

        if not mode_keys_input:
            mode_keys_input.append('1') 

        for mode_key in mode_keys_input:
            if mode_key in self.game_modes:
                mode_data = self.game_modes[mode_key]
                final_kwargs.update(mode_data.get('kwargs', {}))
                effect_names.append(mode_data['name'])
                total_score += mode_data.get('score', 0)
            elif mode_key in self.game_effects:
                effect_data = self.game_effects[mode_key]
                final_kwargs.update(effect_data.get('kwargs', {}))
                effect_names.append(effect_data['name'])
                total_score += effect_data.get('score', 0)
        
        target_song = None
        if song_query.isdigit():
            target_song = next((s for s in self.song_data if s['id'] == int(song_query)), None)
        else:
            found_songs = [s for s in self.song_data if song_query.lower() in s['title'].lower()]
            if found_songs:
                exact_match = next((s for s in found_songs if s['title'].lower() == song_query.lower()), None)
                target_song = exact_match or min(found_songs, key=lambda s: len(s['title']))
        
        if not target_song:
            await event.send(event.plain_result(f'未在数据库中找到与 "{song_query}" 匹配的歌曲。'))
            return

        final_kwargs['force_song_object'] = target_song

        try:
            game_data = await self.start_new_game(**final_kwargs)
        except Exception as e:
            logger.error(f"测试模式下，执行 start_new_game 失败: {e}", exc_info=True)
            game_data = None

        if not game_data:
            await event.send(event.plain_result("......生成测试游戏失败，请检查日志。"))
            return

        correct_song = game_data['song']
        other_songs = random.sample([s for s in self.song_data if s['id'] != correct_song['id']], 11)
        options = [correct_song] + other_songs
        random.shuffle(options)
        correct_answer_num = options.index(correct_song) + 1
        options_img_path = await self._create_options_image(options)
        
        applied_effects = "、".join(effect_names)
        intro_text = f"--- 调试模式 ---\n歌曲: {correct_song['title']}\n效果: {applied_effects}\n答案: {correct_answer_num}"
        
        msg_chain = [Comp.Plain(intro_text)]
        if options_img_path:
            msg_chain.append(Comp.Image(file=options_img_path))
        
        await event.send(event.chain_result(msg_chain))
        await event.send(event.chain_result([Comp.Record(file=game_data["clip_path"])]))

        jacket_source = self._get_resource_path_or_url(f"music_jacket/{correct_song['jacketAssetbundleName']}.png")
        answer_msg = [Comp.Plain(f"[测试模式] 正确答案是: {correct_answer_num}. {correct_song['title']}\n")]
        if jacket_source:
            answer_msg.append(Comp.Image(file=str(jacket_source)))
        await event.send(event.chain_result(answer_msg))

    # --- 统一的听歌功能处理器 ---
    async def _handle_listen_command(self, event: AstrMessageEvent, mode: str):
        """
        统一处理所有"听歌"类指令（钢琴、伴奏、人声等）的通用逻辑。
        :param mode: 'piano', 'accompaniment', 'vocals', 'bass', 'drums'
        """
        # 1. 根据模式确定所需的资源和配置
        mode_config = {
            'piano': {
                'available_songs': self.available_piano_songs,
                'resource_folder': "songs_piano_trimmed_mp3",
                'not_found_msg': "......抱歉，没有找到任何预生成的钢琴曲。",
                'no_match_msg': "......没有找到与 '{search_term}' 匹配的歌曲，或者该歌曲没有可用的钢琴版本。",
                'title_suffix': "(钢琴)",
                'is_piano': True
            },
            'accompaniment': {
                'available_songs': self.available_accompaniment_songs,
                'resource_folder': "accompaniment",
                'not_found_msg': "......抱歉，没有找到任何预生成的伴奏曲。",
                'no_match_msg': "......没有找到与 '{search_term}' 匹配的歌曲，或者该歌曲没有可用的伴奏版本。",
                'title_suffix': "(伴奏)",
                'is_piano': False
            },
            'vocals': {
                'available_songs': self.available_vocals_songs,
                'resource_folder': "vocals_only",
                'not_found_msg': "......抱歉，没有找到任何预生成的纯人声曲。",
                'no_match_msg': "......没有找到与 '{search_term}' 匹配的歌曲，或者该歌曲没有可用的人声版本。",
                'title_suffix': "(人声)",
                'is_piano': False
            },
            'bass': {
                'available_songs': self.available_bass_songs,
                'resource_folder': "bass_only",
                'not_found_msg': "......抱歉，没有找到任何预生成的纯贝斯曲。",
                'no_match_msg': "......没有找到与 '{search_term}' 匹配的歌曲，或者该歌曲没有可用的贝斯版本。",
                'title_suffix': "(贝斯)",
                'is_piano': False
            },
            'drums': {
                'available_songs': self.available_drums_songs,
                'resource_folder': "drums_only",
                'not_found_msg': "......抱歉，没有找到任何预生成的纯鼓点曲。",
                'no_match_msg': "......没有找到与 '{search_term}' 匹配的歌曲，或者该歌曲没有可用的鼓点版本。",
                'title_suffix': "(鼓组)",
                'is_piano': False
            }
        }
        
        config = mode_config[mode]
        
        # 2. 通用的前置条件检查
        if not self._is_group_allowed(event):
            return

        session_id = event.unified_msg_origin
        lock = self.context.game_session_locks.setdefault(session_id, asyncio.Lock())
        
        async with lock:
            cooldown = self.config.get("game_cooldown_seconds", 60)
            if time.time() - self.last_game_end_time.get(session_id, 0) < cooldown:
                remaining_time = cooldown - (time.time() - self.last_game_end_time.get(session_id, 0))
                time_display = f"{remaining_time:.3f}" if remaining_time < 1 else str(int(remaining_time))
                yield event.plain_result(f"嗯......休息 {time_display} 秒再玩吧......")
                return
                
            if session_id in self.context.active_game_sessions:
                yield event.plain_result("......有一个正在进行的游戏或播放任务了呢。")
                return

            user_id = event.get_sender_id()
            loop = asyncio.get_running_loop()
            can_listen = await loop.run_in_executor(self.executor, self._can_listen_song_sync, user_id)
            if not can_listen:
                limit = self.config.get('daily_listen_limit', 5)
                yield event.plain_result(f"......你今天听歌的次数已达上限（{limit}次），请明天再来吧......")
                return
            
            if not config['available_songs']:
                yield event.plain_result(config['not_found_msg'])
                return
            
            # 所有检查通过后，在锁内标记会话
            self.context.active_game_sessions.add(session_id)

        try:
            # --- 新增：为听歌模式发送统计信标 ---
            asyncio.create_task(self._api_ping(f"listen_{mode}"))

            # --- 3. 通用的参数解析和歌曲查找（已修正逻辑） ---
            args = event.message_str.strip().split(maxsplit=1)
            search_term = args[1] if len(args) > 1 else None
            song_to_play = None

            if search_term:
                if search_term.isdigit():
                    # 如果是数字，只按ID搜索
                    music_id_to_find = int(search_term)
                    song_to_play = next((s for s in config['available_songs'] if s['id'] == music_id_to_find), None)
                else:
                    # 否则，按标题搜索
                    found_songs = [s for s in config['available_songs'] if search_term.lower() in s['title'].lower()]
                    if found_songs:
                        exact_match = next((s for s in found_songs if s['title'].lower() == search_term.lower()), None)
                        if exact_match:
                            song_to_play = exact_match
                        else:
                            # 选择最接近的匹配（最短的标题）
                            song_to_play = min(found_songs, key=lambda s: len(s['title']))
                
                if not song_to_play:
                    yield event.plain_result(config['no_match_msg'].format(search_term=search_term))
                    return
            else:
                song_to_play = random.choice(config['available_songs'])
                
            # 4. 通用的会话处理和消息发送
            song = song_to_play
            mp3_source: Optional[Union[Path, str]] = None
            
            # 根据模式确定文件路径（已支持远程资源）
            if config['is_piano']:
                # --- 修正：采用与start_new_game相同的精确匹配逻辑 ---
                all_song_bundles = {v['vocalAssetbundleName'] for v in song.get('vocals', [])}
                valid_piano_bundles = list(all_song_bundles.intersection(self.available_piano_songs_bundles))

                if valid_piano_bundles:
                    chosen_bundle = random.choice(valid_piano_bundles)
                    relative_path = f"songs_piano_trimmed_mp3/{chosen_bundle}/{chosen_bundle}.mp3"
                    mp3_source = self._get_resource_path_or_url(relative_path)
            else:
                sekai_ver = next((v for v in song.get('vocals', []) if v.get('musicVocalType') == 'sekai'), None)
                bundle_name = None
                if sekai_ver:
                    bundle_name = sekai_ver.get('vocalAssetbundleName')
                elif song.get('vocals'):
                    bundle_name = song['vocals'][0].get('vocalAssetbundleName')
                
                if bundle_name and bundle_name in self.preprocessed_tracks[config['resource_folder']]:
                     relative_path = f"{config['resource_folder']}/{bundle_name}.mp3"
                     mp3_source = self._get_resource_path_or_url(relative_path)

            if not mp3_source:
                logger.error(f"逻辑错误：歌曲 {song['title']} 在可用列表但找不到其有效的MP3文件（模式: {mode}）。")
                yield event.plain_result("......出错了，找不到有效的音频文件。")
                return

            # 图片部分保持本地逻辑，按您的要求不作改动
            jacket_source = self._get_resource_path_or_url(f"music_jacket/{song['jacketAssetbundleName']}.png")
            
            msg_chain = [Comp.Plain(f"歌曲:{song['id']}. {song['title']} {config['title_suffix']}\n")]
            if jacket_source:
                msg_chain.append(Comp.Image(file=str(jacket_source)))
            
            yield event.chain_result(msg_chain)
            yield event.chain_result([Comp.Record(file=str(mp3_source))])

            user_id = event.get_sender_id()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self.executor, self._record_listen_song, user_id, event.get_sender_name(), session_id)
            
            # --- 新增：为听歌功能发送统计日志 ---
            asyncio.create_task(self._api_log_game({
                "game_type": 'listen',
                "game_mode": mode,
                "user_id": user_id,
                "user_name": event.get_sender_name(),
                "is_correct": False,
                "score_awarded": 0, # 修正：字段名以匹配服务器
                "session_id": session_id
            }))

            self.last_game_end_time[session_id] = time.time()

        except Exception as e:
            logger.error(f"处理听歌功能(模式: {mode})时出错: {e}", exc_info=True)
            yield event.plain_result("......播放时出错了，请联系管理员。")
        finally:
            if session_id in self.context.active_game_sessions:
                self.context.active_game_sessions.remove(session_id)


    @filter.command("听钢琴", alias={"listen_piano"})
    async def listen_to_piano(self, event: AstrMessageEvent):
        """随机或指定播放一首预生成的钢琴曲"""
        async for result in self._handle_listen_command(event, mode='piano'):
            yield result

    @filter.command("听伴奏", alias={"listen_karaoke"})
    async def listen_to_accompaniment(self, event: AstrMessageEvent):
        """随机或指定播放一首预生成的纯伴奏曲"""
        async for result in self._handle_listen_command(event, mode='accompaniment'):
            yield result

    @filter.command("听人声", alias={"listen_vocals"})
    async def listen_to_vocals(self, event: AstrMessageEvent):
        """随机或指定播放一首预生成的纯人声曲"""
        async for result in self._handle_listen_command(event, mode='vocals'):
            yield result

    @filter.command("听贝斯", alias={"listen_bass"})
    async def listen_to_bass(self, event: AstrMessageEvent):
        """随机或指定播放一首预生成的纯贝斯曲"""
        async for result in self._handle_listen_command(event, mode='bass'):
            yield result

    @filter.command("听鼓组", alias={"listen_drums"})
    async def listen_to_drums(self, event: AstrMessageEvent):
        """听鼓组音轨"""
        async for result in self._handle_listen_command(event, mode="drums"):
            yield result

    # ... (在 listen_to_drums 方法之后，listen_to_another_vocal 之前，添加这个新函数) ...
    def _find_song_by_query(self, query: str) -> Optional[Dict]:
        """通过ID或名称统一查找歌曲，优先精确匹配。"""
        if query.isdigit():
            return next((s for s in self.song_data if s['id'] == int(query)), None)
        else:
            query_lower = query.lower()
            found_songs = [s for s in self.song_data if query_lower in s['title'].lower()]
            if not found_songs:
                return None
            
            exact_match = next((s for s in found_songs if s['title'].lower() == query_lower), None)
            return exact_match or min(found_songs, key=lambda s: len(s['title']))

    async def _handle_list_anov_versions(self, event: AstrMessageEvent, song: Dict):
        """[辅助函数] 当用户仅提供歌曲时，列出所有可用的ANOV版本。"""
        anov_list = [v for v in song.get('vocals', []) if v.get('musicVocalType') == 'another_vocal']
        if not anov_list:
            yield event.plain_result(f"......歌曲 '{song['title']}' 没有 Another Vocal 版本。")
            return

        reply = f"歌曲 \"{song['title']}\" 有以下 Another Vocal 版本:\n"
        lines = []
        for v in anov_list:
            names = [self.character_data.get(str(c['characterId']), {}).get('fullName', '未知') for c in v.get('characters', [])]
            abbrs = [self.character_data.get(str(c['characterId']), {}).get('name', 'unk') for c in v.get('characters', [])]
            lines.append(f"  - {' + '.join(names)} ({'+'.join(abbrs)})")
        reply += "\n".join(lines)
        reply += f"\n\n请使用 /听anov {song['id']} <角色> 来播放。"
        yield event.plain_result(reply)

    @filter.command("听anov", alias={"listen_anov", "listen_another_vocal", "anov"})
    async def listen_to_another_vocal(self, event: AstrMessageEvent):
        """听指定歌曲的 another vocal 版本。支持多种用法。"""
        # 1. 通用的前置条件检查
        if not self._is_group_allowed(event): return
        session_id = event.unified_msg_origin
        lock = self.context.game_session_locks.setdefault(session_id, asyncio.Lock())
        async with lock:

            cooldown = self.config.get("game_cooldown_seconds", 60)
            if time.time() - self.last_game_end_time.get(session_id, 0) < cooldown:
                yield event.plain_result(f"嗯......休息 {cooldown - (time.time() - self.last_game_end_time.get(session_id, 0)):.1f} 秒再玩吧......")
                return
            if session_id in self.context.active_game_sessions:
                yield event.plain_result("......有一个正在进行的游戏或播放任务了呢。")
                return

            user_id = event.get_sender_id()
            can_listen = await asyncio.get_running_loop().run_in_executor(self.executor, self._can_listen_song_sync, user_id)
            if not can_listen:
                limit = self.config.get('daily_listen_limit', 5)
                yield event.plain_result(f"......你今天听歌的次数已达上限（{limit}次），请明天再来吧......")
                return
            
            if not self.another_vocal_songs:
                yield event.plain_result("......抱歉，没有找到任何可用的 Another Vocal 歌曲。")
                return
            
            self.context.active_game_sessions.add(session_id)

        try:
            await asyncio.create_task(self._api_ping("listen_another_vocal"))
            
            raw_content = event.message_str.strip().split(maxsplit=1)
            content = raw_content[1] if len(raw_content) > 1 else ""

            song_to_play, vocal_info = None, None

            # Case 1: `听anov` (无参数)
            if not content:
                song_to_play = random.choice(self.another_vocal_songs)
                anov_list = [v for v in song_to_play.get('vocals', []) if v.get('musicVocalType') == 'another_vocal']
                if anov_list: vocal_info = random.choice(anov_list)
            else:
                parts = content.rsplit(maxsplit=1)
                last_part = parts[-1].lower()
                
                # 尝试将最后一个词解析为角色组合
                is_char_combo = True
                target_ids = set()
                for abbr in last_part.split('+'):
                    char_id = self.abbr_to_char_id.get(abbr)
                    if char_id is None:
                        is_char_combo = False
                        break
                    target_ids.add(char_id)
                
                # Case 4: `... <歌曲> <角色>`
                if is_char_combo and len(parts) > 1:
                    song_query = parts[0]
                    song_to_play = self._find_song_by_query(song_query)
                    if not song_to_play:
                        yield event.plain_result(f"......没有找到歌曲 '{song_query}'。")
                        return
                    for v in song_to_play.get('vocals', []):
                        if v.get('musicVocalType') == 'another_vocal' and {c.get('characterId') for c in v.get('characters', [])} == target_ids:
                            vocal_info = v
                            break
                    if not vocal_info:
                        yield event.plain_result(f"......歌曲 '{song_to_play['title']}' 没有 '{last_part}' 的 Another Vocal 版本。")
                        return
                else:
                    # Case 2: `... <角色>`
                    if len(parts) == 1 and is_char_combo and len(target_ids) == 1:
                        char_id = list(target_ids)[0]
                        songs_by_char = self.char_id_to_anov_songs.get(char_id)
                        if not songs_by_char:
                            char_name = self.character_data.get(str(char_id), {}).get("fullName", content)
                            yield event.plain_result(f"......抱歉，没有找到 {char_name} 的 Another Vocal 歌曲。")
                            return
                        song_to_play = random.choice(songs_by_char)
                        solo = next((v for v in song_to_play.get('vocals', []) if v.get('musicVocalType') == 'another_vocal' and len(v.get('characters',[])) == 1 and v['characters'][0].get('characterId') == char_id), None)
                        vocal_info = solo or next((v for v in song_to_play.get('vocals', []) if v.get('musicVocalType') == 'another_vocal' and any(c.get('characterId') == char_id for c in v.get('characters', []))), None)
                    # Case 3: `... <歌曲>`
                    else:
                        song_to_play = self._find_song_by_query(content)
                        if not song_to_play:
                            yield event.plain_result(f"......没有找到与 '{content}' 匹配的歌曲或角色。")
                            return
                        async for result in self._handle_list_anov_versions(event, song_to_play):
                            yield result
                        return

            if not song_to_play or not vocal_info:
                yield event.plain_result("......内部错误，请联系管理员。")
                return
            
            # 后续流程...
            char_ids = [c.get('characterId') for c in vocal_info.get('characters', [])]
            char_id_for_cache = '_'.join(map(str, sorted(char_ids)))
            output_filename = f"anov_{song_to_play['id']}_{char_id_for_cache}.mp3"
            output_path = self.output_dir / output_filename

            if not output_path.exists():
                logger.info(f"缓存文件 {output_filename} 不存在，正在创建...")
                mp3_source = self._get_resource_path_or_url(f"songs/{vocal_info['vocalAssetbundleName']}/{vocal_info['vocalAssetbundleName']}.mp3")
                if not mp3_source:
                    yield event.plain_result("......出错了，找不到有效的音频文件。")
                    return
                filler_sec = song_to_play.get('fillerSec', 0)
                command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-ss', str(filler_sec), '-i', str(mp3_source), '-c:a', 'copy', '-f', 'mp3', str(output_path)]
                proc = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                _, stderr = await proc.communicate()
                if proc.returncode != 0:
                    logger.error(f"FFmpeg failed. Stderr: {stderr.decode(errors='ignore')}")
                    if output_path.exists(): os.remove(output_path)
                    yield event.plain_result("......处理音频时出错了（FFmpeg）。")
                    return
            else:
                logger.info(f"使用已缓存的文件: {output_filename}")

            jacket_source = self._get_resource_path_or_url(f"music_jacket/{song_to_play['jacketAssetbundleName']}.png")
            char_names = [self.character_data.get(str(cid), {}).get('fullName', '未知') for cid in char_ids]
            
            msg_chain = [Comp.Plain(f"歌曲:{song_to_play['id']}. {song_to_play['title']} (Another Vocal - {' + '.join(char_names)})\n")]
            if jacket_source: msg_chain.append(Comp.Image(file=str(jacket_source)))
            
            yield event.chain_result(msg_chain)
            yield event.chain_result([Comp.Record(file=str(output_path))])

            user_id = event.get_sender_id()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self.executor, self._record_listen_song, user_id, event.get_sender_name(), session_id)
            await asyncio.create_task(self._api_log_game({"game_type": 'listen', "game_mode": 'another_vocal', "user_id": user_id, "user_name": event.get_sender_name(), "is_correct": False, "score_awarded": 0, "session_id": session_id}))
            self.last_game_end_time[session_id] = time.time()

        except Exception as e:
            logger.error(f"处理听anov功能时出错: {e}", exc_info=True)
            yield event.plain_result("......播放时出错了，请联系管理员。")
        finally:
            if session_id in self.context.active_game_sessions:
                self.context.active_game_sessions.remove(session_id)

    def _get_all_user_stats_sync(self):
        """获取所有用户的统计数据以用于迁移。"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id, user_name, score FROM user_stats WHERE score > 0")
            return cursor.fetchall()

    def _get_all_mode_stats_sync(self):
        """新增：获取所有题型统计数据以用于迁移。"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT mode, total_attempts, correct_attempts FROM mode_stats")
            return cursor.fetchall()

    @filter.command("同步分数", alias={"syncscore", "migrategs"})
    async def sync_scores_to_server(self, event: AstrMessageEvent):
        """（管理员）将所有用户的本地总分同步到服务器。"""
        if str(event.get_sender_id()) not in self.config.get("super_users", []):
            yield event.plain_result("......权限不足，只有管理员才能执行此操作。")
            return

        if not self.api_key:
            yield event.plain_result("......未配置服务器排行榜功能，无法同步。请先在配置文件中设置API密钥。")
            return

        if not self.stats_server_url:
            yield event.plain_result("......服务器地址配置不正确，无法同步。")
            return
        
        yield event.plain_result("......正在准备同步所有本地玩家分数至服务器排行榜...")

        loop = asyncio.get_running_loop()
        all_local_users = await loop.run_in_executor(self.executor, self._get_all_user_stats_sync)
        
        if not all_local_users:
            yield event.plain_result("......本地没有任何玩家数据，无需同步。")
            return

        payload = [
            {"user_id": str(user[0]), "user_name": user[1], "score": user[2]}
            for user in all_local_users
        ]

        migrate_url = f"{self.stats_server_url}/api/migrate_leaderboard"
        try:
            session = await self._get_session()
            if not session:
                yield event.plain_result("......网络组件初始化失败。")
                return
            
            yield event.plain_result(f"......正在将 {len(payload)} 条玩家数据同步至服务器...")
            async with session.post(migrate_url, json=payload, headers=self._get_api_headers(), timeout=60) as response:
                if response.status == 200:
                    result = await response.json()
                    yield event.plain_result(f"✅ 分数同步成功！\n处理了 {result.get('processed_count', 0)} 条记录。\n新增了 {result.get('new_records', 0)} 条记录。\n更新了 {result.get('updated_records', 0)} 条记录。")
                elif response.status == 401:
                    yield event.plain_result(f"❌ 分数同步失败：API密钥无效。")
                else:
                    yield event.plain_result(f"❌ 分数同步失败，服务器返回错误：{response.status} {await response.text()}")

        except Exception as e:
            logger.error(f"同步服务器分数失败: {e}", exc_info=True)
            yield event.plain_result(f"❌ 同步失败，发生网络错误或服务器无响应。")

    def _can_listen_song_sync(self, user_id: str) -> bool:
        """[Helper] 同步检查听歌次数"""
        daily_limit = self.config.get("daily_listen_limit", 5)
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT daily_listen_songs, last_listen_date FROM user_stats WHERE user_id = ?", (user_id,))
            user_data = cursor.fetchone()
            return not (user_data and user_data[1] == time.strftime("%Y-%m-%d") and user_data[0] >= daily_limit)


    def _get_user_local_global_stats_sync(self, user_id: str) -> Optional[Dict]:
        """[辅助函数] 同步获取用户的本地全局统计数据（用于备用和每日次数）。"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT score, attempts, correct_attempts, daily_games_played, last_played_date FROM user_stats
                WHERE user_id = ?
            """, (user_id,))
            global_row = cursor.fetchone()
            
            if not global_row:
                return None

            global_score = global_row[0]
            cursor.execute("SELECT COUNT(*) + 1 FROM user_stats WHERE score > ?", (global_score,))
            global_rank = cursor.fetchone()[0]

            return {
                'score': global_row[0],
                'attempts': global_row[1],
                'correct': global_row[2],
                'daily_plays': global_row[3],
                'last_play_date': global_row[4],
                'rank': global_rank
            }


    # --- 新增：API 通信模块 ---
    def _get_stats_server_root(self) -> Optional[str]:
        """根据配置获取统计服务器的根URL。"""
        url_base = self.config.get("remote_resource_url_base", "")
        if not url_base:
            logger.warning("API密钥已配置, 但 'remote_resource_url_base' 未设置, 无法确定统计服务器地址。")
            return None
        parsed_url = urlparse(url_base)
        return f"{parsed_url.scheme}://{parsed_url.hostname}:5000"

    def _get_api_headers(self) -> Dict[str, str]:
        """获取带有认证信息的API请求头。"""
        return {"X-API-KEY": self.api_key} if self.api_key else {}

    async def _api_ping(self, event_type: str):
        """向服务器发送一个简单的事件埋点。"""
        if not self.stats_server_url: return
        
        ping_url = f"{self.stats_server_url}/api/ping/{event_type}" # 修正：添加 /api 前缀
        try:
            session = await self._get_session()
            if not session: return
            async with session.get(ping_url, headers=self._get_api_headers(), timeout=2):
                pass
        except Exception as e:
            logger.warning(f"Stats ping to {ping_url} failed: {e}")

    async def _api_log_game(self, game_log_data: dict):
        """向服务器记录一条详细的游戏日志。"""
        if not self.stats_server_url: return

        post_url = f"{self.stats_server_url}/api/log_game" # 修正：添加 /api 前缀
        try:
            session = await self._get_session()
            if not session: return
            async with session.post(post_url, json=game_log_data, headers=self._get_api_headers(), timeout=3) as resp:
                if resp.status != 200:
                    logger.warning(f"记录游戏日志失败. Status: {resp.status}, Response: {await resp.text()}")
        except Exception as e:
            logger.warning(f"发送游戏日志至 {post_url} 失败: {e}")

    async def _api_update_score(self, user_id: str, user_name: str, score_delta: int):
        """向服务器同步玩家的分数变化。"""
        if not self.stats_server_url or score_delta == 0:
            return

        payload = {
            "user_id": str(user_id),
            "user_name": user_name,
            "score_change": score_delta # 修正：字段名以匹配服务器
        }
        post_url = f"{self.stats_server_url}/api/update_score" # 修正：添加 /api 前缀
        try:
            session = await self._get_session()
            if not session: return
            async with session.post(post_url, json=payload, headers=self._get_api_headers(), timeout=3) as resp:
                 if resp.status != 200:
                    logger.warning(f"同步分数至服务器失败. Status: {resp.status}, Response: {await resp.text()}")
        except Exception as e:
            logger.warning(f"发送分数更新至 {post_url} 失败: {e}")


    async def _api_get_user_global_stats(self, user_id: str) -> Optional[Dict]:
        """通过API获取用户的服务器统计数据。"""
        if not self.stats_server_url:
            return None

        stats_url = f"{self.stats_server_url}/api/user_stats/{user_id}"
        try:
            session = await self._get_session()
            if not session:
                logger.warning("aiohttp session 不可用，无法获取服务器用户数据。")
                return None
            async with session.get(stats_url, headers=self._get_api_headers(), timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"成功从API获取用户 {user_id} 的服务器数据: {data}")
                    return data
                elif response.status == 404:
                    logger.info(f"用户 {user_id} 在服务器排行榜上尚无数据。")
                    return None 
                else:
                    logger.warning(f"获取用户 {user_id} 服务器数据失败. Status: {response.status}, Response: {await response.text()}")
                    return None
        except Exception as e:
            logger.error(f"请求 {stats_url} 失败: {e}", exc_info=True)
            return None

    def _get_full_group_ranking_sync(self, session_id: str) -> List[Tuple]:
        """
        [核心] 从本地数据库获取指定群聊的完整排行榜。
        - 从 user_stats 表中筛选出所有在该群聊中有记录的用户。
        - 解析每个用户的 group_scores JSON。
        - 提取该群聊的分数、尝试次数、正确次数。
        - 过滤掉分数为0的玩家。
        - 按分数降序排序后返回。
        """
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_id, user_name, group_scores 
                FROM user_stats
            """)
            all_users_data = cursor.fetchall()

            group_ranking = []
            for user_id, user_name, group_scores_json in all_users_data:
                if not group_scores_json:
                    continue
                
                try:
                    group_scores = json.loads(group_scores_json)
                    group_stat_raw = group_scores.get(session_id)
                    
                    if not group_stat_raw:
                        continue
                    
                    # [兼容处理] 兼容旧的纯分数格式和新的字典格式
                    if isinstance(group_stat_raw, int):
                        score = group_stat_raw
                        attempts, correct_attempts = -1, -1 # 旧数据标记为不可用
                    elif isinstance(group_stat_raw, dict):
                        score = group_stat_raw.get("score", 0)
                        attempts = group_stat_raw.get("attempts", 0)
                        correct_attempts = group_stat_raw.get("correct_attempts", 0)
                    else:
                        continue # 跳过无法识别的格式

                    if score > 0:
                        group_ranking.append((user_id, user_name, score, attempts, correct_attempts))

                except (json.JSONDecodeError, TypeError):
                    continue
        
        group_ranking.sort(key=lambda x: x[2], reverse=True)
        return group_ranking

    def _get_user_stats_in_group_sync(self, user_id_to_find: str, session_id: str) -> Optional[Dict]:
        """
        [新] 使用统一数据源获取单个用户在群组中的分数、排名和详细统计。
        """
        full_ranking = self._get_full_group_ranking_sync(session_id)
        
        # 在排行榜（分数>0）中查找用户
        for i, (user_id, _, score, attempts, correct_attempts) in enumerate(full_ranking):
            if user_id == user_id_to_find:
                return {
                    "score": score, 
                    "rank": i + 1,
                    "attempts": attempts,
                    "correct_attempts": correct_attempts
                }
        
        # 如果用户不在排行榜上（分数为0或无记录），则手动查找
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT group_scores FROM user_stats WHERE user_id = ?", (user_id_to_find,))
            row = cursor.fetchone()
            if row and row[0]:
                try:
                    group_scores = json.loads(row[0])
                    group_stat = group_scores.get(session_id)
                    if isinstance(group_stat, dict):
                        return {
                            "score": group_stat.get("score", 0), 
                            "rank": None,
                            "attempts": group_stat.get("attempts", 0),
                            "correct_attempts": group_stat.get("correct_attempts", 0)
                        }
                    elif isinstance(group_stat, int): # 兼容旧数据
                        return {"score": group_stat, "rank": None, "attempts": -1, "correct_attempts": -1}

                except (json.JSONDecodeError, TypeError):
                    pass
        
        # 对于全新用户或无任何记录的用户
        return {"score": 0, "rank": None, "attempts": 0, "correct_attempts": 0}

    def _get_user_local_global_stats_sync(self, user_id: str) -> Optional[Dict]:
        """[辅助函数] 同步获取用户的本地全局统计数据（用于备用和每日次数）。"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT score, attempts, correct_attempts, daily_games_played, last_played_date FROM user_stats
                WHERE user_id = ?
            """, (user_id,))
            global_row = cursor.fetchone()
            
            if not global_row:
                return None

            global_score = global_row[0]
            cursor.execute("SELECT COUNT(*) + 1 FROM user_stats WHERE score > ?", (global_score,))
            global_rank = cursor.fetchone()[0]

            return {
                'score': global_row[0],
                'attempts': global_row[1],
                'correct': global_row[2],
                'daily_plays': global_row[3],
                'last_play_date': global_row[4],
                'rank': global_rank
            }

    def _precompute_random_combinations(self) -> Dict[int, List[Dict]]:
        """
        [新] 预计算所有可行的随机效果组合。
        该方法会检查资源可用性，并处理好多效果组合的特殊规则（如1.5倍速）。
        返回一个按最终分数分组的字典。
        """
        combinations_by_score = defaultdict(list)

        # 1. 筛选出当前可玩的"音源类"效果
        playable_source_effects = []
        for effect in self.source_effects:
            kwargs = effect.get('kwargs', {})
            if 'play_preprocessed' in kwargs:
                mode = kwargs['play_preprocessed']
                if self.preprocessed_tracks.get(mode):
                    playable_source_effects.append(effect)
            elif 'melody_to_piano' in kwargs:
                if self.available_piano_songs:
                    playable_source_effects.append(effect)
            else:  # 假设普通模式总是可用
                playable_source_effects.append(effect)

        # 2. 构建"独立类"效果的选项（开启/关闭）
        independent_options = []
        # 如果启用了轻量模式，则不使用变速、倒放等效果
        active_base_effects = [] if self.lightweight_mode else self.base_effects
        for effect in active_base_effects:
            # (开启效果, 关闭效果)
            independent_options.append([effect, {'name': 'Off', 'score': 0, 'kwargs': {}}])

        if not playable_source_effects:
            return {}

        # 3. 使用 itertools.product 枚举所有组合
        for source_effect in playable_source_effects:
            for independent_choices in itertools.product(*independent_options):
                

                # 新增约束：钢琴模式和倒放模式不能同时出现
                is_piano_mode = 'melody_to_piano' in source_effect.get('kwargs', {})
                has_reverse_effect = any('reverse_audio' in choice.get('kwargs', {}) for choice in independent_choices)
                if is_piano_mode and has_reverse_effect:
                    continue
                
                raw_combination = [source_effect] + [choice for choice in independent_choices if choice['score'] > 0]
                
                final_effects_list = []
                final_kwargs = {}
                base_score = 0
                
                # 特殊处理：当效果多于1个时，2倍速降为1.5倍速
                is_multi_effect = len(raw_combination) > 1
                
                for effect_template in raw_combination:
                    # 深拷贝以避免修改类属性中的原始定义
                    effect = {k: (v.copy() if isinstance(v, dict) else v) for k, v in effect_template.items()}
                    
                    if is_multi_effect and 'speed_multiplier' in effect.get('kwargs', {}):
                        effect['kwargs']['speed_multiplier'] = 1.5
                        effect['name'] = '1.5倍速'
                    
                    final_effects_list.append(effect)
                    final_kwargs.update(effect.get('kwargs', {}))
                    base_score += effect.get('score', 0)

                # 4. 计算最终分数（包含组合加分）
                final_score = base_score
                if 'speed_multiplier' in final_kwargs and 'reverse_audio' in final_kwargs:
                    final_score += 1

                # 5. 将处理好的完整信息存入字典
                processed_combo = {
                    'effects_list': final_effects_list,
                    'final_kwargs': final_kwargs,
                    'final_score': final_score,
                }
                combinations_by_score[final_score].append(processed_combo)
                
        return dict(combinations_by_score)

    def _get_random_target_distribution(self, combinations_by_score: Dict[int, list]) -> Dict[int, float]:
        """
        [新] 根据预计算的组合和衰减因子，生成目标分数概率分布。
        """
        if not combinations_by_score:
            return {}

        scores = sorted(combinations_by_score.keys())
        decay_factor = self.random_mode_decay_factor
        
        # 使用几何衰减模型计算每个分数的权重
        weights = [decay_factor ** score for score in scores]
        
        # 归一化权重以得到概率
        total_weight = sum(weights)
        if total_weight == 0: # 避免除以零
            # 如果权重和为0，回退到均匀分布
            return {score: 1.0 / len(scores) for score in scores}
            
        probabilities = [w / total_weight for w in weights]
        
        return dict(zip(scores, probabilities))

    def _draw_help_image_sync(self) -> Optional[str]:
        """[辅助函数] 同步绘制帮助图片。"""
        try:
            width, height = 800, 1350
            bg_color_start, bg_color_end = (230, 240, 255), (200, 210, 240)
            img = Image.new("RGB", (width, height), bg_color_start)
            draw_bg = ImageDraw.Draw(img)
            for y in range(height):
                r = int(bg_color_start[0] + (bg_color_end[0] - bg_color_start[0]) * y / height)
                g = int(bg_color_start[1] + (bg_color_end[1] - bg_color_start[1]) * y / height)
                b = int(bg_color_start[2] + (bg_color_end[2] - bg_color_start[2]) * y / height)
                draw_bg.line([(0, y), (width, y)], fill=(r, g, b))

            background_path = self.resources_dir / "ranking_bg.png"
            if background_path.exists():
                try:
                    custom_bg = Image.open(background_path).convert("RGBA").resize((width, height), LANCZOS)
                    custom_bg.putalpha(128)
                    img = img.convert("RGBA")
                    img = Image.alpha_composite(img, custom_bg)
                except Exception as e:
                    logger.warning(f"加载或混合自定义背景图片失败: {e}")

            if img.mode != 'RGBA': img = img.convert('RGBA')
            white_overlay = Image.new("RGBA", img.size, (255, 255, 255, 100))
            img = Image.alpha_composite(img, white_overlay)

            font_color, shadow_color = (30, 30, 50), (180, 180, 190, 128)
            header_color = (80, 90, 120)
            
            try:
                font_path = str(self.resources_dir / "font.ttf")
                title_font = ImageFont.truetype(font_path, 48)
                section_font = ImageFont.truetype(font_path, 32)
                body_font = ImageFont.truetype(font_path, 24)
                id_font = ImageFont.truetype(font_path, 16)
            except IOError:
                title_font = ImageFont.load_default(size=48)
                section_font = ImageFont.load_default(size=32)
                body_font = ImageFont.load_default(size=24)
                id_font = ImageFont.load_default(size=16)

            help_text = (
                "--- PJSK猜歌插件帮助 ---\n\n"
                "🎵 基础指令\n"
                f"  `猜歌` - {self.game_modes['normal']['name']} ({self.game_modes['normal']['score']}分)\n"
                f"  `猜歌 1` - {self.game_modes['1']['name']} ({self.game_modes['1']['score']}分)\n"
                f"  `猜歌 2` - {self.game_modes['2']['name']} ({self.game_modes['2']['score']}分)\n"
                f"  `猜歌 3` - {self.game_modes['3']['name']} ({self.game_modes['3']['score']}分)\n"
                f"  `猜歌 4` - {self.game_modes['4']['name']} ({self.game_modes['4']['score']}分)\n"
                f"  `猜歌 5` - {self.game_modes['5']['name']} ({self.game_modes['5']['score']}分)\n"
                f"  `猜歌 6` - {self.game_modes['6']['name']} ({self.game_modes['6']['score']}分)\n"
                f"  `猜歌 7` - {self.game_modes['7']['name']} ({self.game_modes['7']['score']}分)\n\n"
                "🎲 高级指令\n"
                "  `随机猜歌` - 随机组合效果 (最高9分)\n"
                "  `猜歌手` - 竞猜演唱者 (测试功能, 不计分)\n"
                "  `听<模式> [歌名/ID]` - 播放指定或随机歌曲的特殊音轨。\n"
                "    可用模式: 钢琴, 伴奏, 人声, 贝斯, 鼓组\n"
                "  `听anov [歌名/ID] [角色名缩写]` - 播放指定或随机的Another\n"
                "   Vocal。可指定角色后随机\n"
                "    (该功能有统一的每日次数限制)\n\n"
                "📊 数据统计\n"
                "  `猜歌分数` - 查看自己的猜歌积分和排名\n"
                "  `群猜歌排行榜` - 查看本群猜歌排行榜\n"
                "  `本地猜歌排行榜` - 查看插件本地存储的猜歌排行榜\n"
                "  `猜歌排行榜` - 查看服务器猜歌总排行榜 (联网)\n"
                "  `同步分数` - (管理员)将本地总分同步至服务器\n"
                "  `查看统计` - 查看各题型的正确率排行"
            )

            with Pilmoji(img) as pilmoji:
                center_x, current_y = width // 2, 80
                x_margin = 60
                line_height_body = 40
                line_height_section = 55
                
                lines = help_text.split('\n')
                
                title_text = lines[0].replace("---", "").strip()
                pilmoji.text((int(center_x) + 2, int(current_y) + 2), title_text, font=title_font, fill=shadow_color, anchor="mm")
                pilmoji.text((int(center_x), int(current_y)), title_text, font=title_font, fill=font_color, anchor="mm")
                current_y += 100

                for line in lines[2:]:
                    if not line.strip():
                        current_y += line_height_body // 2
                        continue

                    if line.startswith("🎵") or line.startswith("🎲") or line.startswith("📊"):
                        font = section_font
                        y_increment = line_height_section
                        text_to_draw = line.strip()
                    else:
                        font = body_font
                        y_increment = line_height_body
                        text_to_draw = line

                    pilmoji.text((x_margin, int(current_y)), text_to_draw, font=font, fill=font_color)
                    current_y += y_increment
                
                footer_text = f"GuessSong v{PLUGIN_VERSION} | Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                pilmoji.text((int(center_x), height - 40), footer_text, font=id_font, fill=header_color, anchor="ms")
            
            img_path = self.output_dir / f"guess_song_help_{int(time.time())}.png"
            img.save(img_path)
            return str(img_path)
        except Exception as e:
            logger.error(f"生成帮助图片时出错: {e}", exc_info=True)
            return None

