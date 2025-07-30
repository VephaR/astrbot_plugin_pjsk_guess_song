import asyncio
import json
import random
import time
import os
import sqlite3
import logging
import re
import subprocess  # 新增
import io
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from pilmoji import Pilmoji
import sys
import traceback
import itertools
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen # 新增
from urllib.error import URLError # 新增
from urllib.parse import urlparse

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

try:
    from astrbot.api import logger
except ImportError:
    logger = logging.getLogger(__name__)

try:
    from astrbot.api.event import filter, AstrMessageEvent
    from astrbot.api.star import Context, Star, register
    import astrbot.api.message_components as Comp
    from astrbot.core.utils.session_waiter import session_waiter, SessionController
    from astrbot.api import AstrBotConfig
except ImportError:
    logger.error("Failed to import from astrbot.api, attempting fallback.")
    from astrbot.core.plugin import Plugin as Star, Context, register, filter, AstrMessageEvent  # type: ignore
    import astrbot.core.message_components as Comp  # type: ignore
    from astrbot.core.utils.session_waiter import session_waiter, SessionController  # type: ignore


# --- 插件元数据 ---
PLUGIN_NAME = "pjsk_guess_song"
PLUGIN_AUTHOR = "nichinichisou"
PLUGIN_DESCRIPTION = "PJSK猜歌插件"
PLUGIN_VERSION = "1.1.0"
PLUGIN_REPO_URL = "https://github.com/nichinichisou0609/astrbot_plugin_pjsk_guess_song"


# --- 数据库管理 ---
def get_db_path(context: Context, plugin_dir: Path) -> str:
    """获取插件的数据库文件路径"""
    config = context.get_config()
    data_path = config.get('data_path')
    if not data_path:
        logger.warning("'data_path' not found in config. Falling back to .../data/ directory.")
        data_path = plugin_dir.parent.parent

    plugin_data_dir = Path(str(data_path)) / 'plugins_data' / PLUGIN_NAME
    os.makedirs(plugin_data_dir, exist_ok=True)
    return str(plugin_data_dir / "guess_song_data.db")


def init_db(db_path: str):
    """初始化数据库表"""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_stats (
                user_id TEXT PRIMARY KEY,
                user_name TEXT,
                score INTEGER DEFAULT 0,
                attempts INTEGER DEFAULT 0,
                correct_attempts INTEGER DEFAULT 0,
                last_play_date TEXT,
                daily_plays INTEGER DEFAULT 0
            )
            """
        )
        # --- 统一的每日听歌次数 ---
        try:
            cursor.execute("ALTER TABLE user_stats ADD COLUMN daily_listen_plays INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass # 字段已存在
            
        # 移除旧的、分离的字段（如果存在）
        # 注意：这在SQLite中比较复杂，通常我们选择不再使用它们
        # 为简单起见，我们将保留旧字段，但代码逻辑不再使用

        # 新增题型统计表
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS mode_stats (
                mode TEXT PRIMARY KEY,
                total_attempts INTEGER DEFAULT 0,
                correct_attempts INTEGER DEFAULT 0
            )
            """
        )
        conn.commit()


# --- 数据加载 ---
def load_song_data(resources_dir: Path) -> Optional[List[Dict]]:
    """加载 guess_song.json 数据"""
    try:
        songs_file = resources_dir / "guess_song.json"
        with open(songs_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        logger.error(f"加载歌曲数据失败: {e}. 请确保 'guess_song.json' 和 'musicVocals.json' 在 'resources' 目录中。")
        return None


def load_character_data(resources_dir: Path) -> Dict[str, str]:
    """加载 characters.json 数据"""
    try:
        char_file = resources_dir / "characters.json"
        with open(char_file, "r", encoding="utf-8") as f:
            chars = json.load(f)
            return {str(c['characterId']): c['name'] for c in chars}
    except FileNotFoundError:
        logger.error("加载角色数据失败: 'characters.json' 未找到。")
        return {}


# --- 核心插件类 ---
@register(PLUGIN_NAME, PLUGIN_AUTHOR, PLUGIN_DESCRIPTION, PLUGIN_VERSION, PLUGIN_REPO_URL)
class GuessSongPlugin(Star):  # type: ignore
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.plugin_dir = Path(os.path.dirname(__file__))
        self.resources_dir = self.plugin_dir / "resources"
        self.output_dir = self.plugin_dir / "output"
        self.db_path = get_db_path(context, self.plugin_dir)
        init_db(self.db_path)
        self.songs_data = load_song_data(self.resources_dir)
        self.character_map = load_character_data(self.resources_dir)
        self.last_game_end_time = {}
        self.available_piano_songs = []
        self.available_accompaniment_songs = []
        self.available_vocals_songs = []
        self.available_bass_songs = []
        self.available_drums_songs = []
        self.bundle_to_song_map = {}

        # --- 新增：为所有固定游戏模式创建配置映射 ---
        self.game_modes = {
            # Key `None` for the base command '猜歌'
            None: {'kwargs': {'score': 1}},
            "1": {'kwargs': {'speed_multiplier': 2.0, 'score': 1}},
            "2": {'kwargs': {'reverse_audio': True, 'score': 3}},
            "3": {'kwargs': {'melody_to_piano': True, 'score': 2}},
            "4": {'kwargs': {'play_preprocessed': 'accompaniment', 'score': 1}},
            "5": {'kwargs': {'play_preprocessed': 'bass_only', 'score': 3}},
            "6": {'kwargs': {'play_preprocessed': 'drums_only', 'score': 4}},
            "7": {'kwargs': {'play_preprocessed': 'vocals_only', 'score': 1}},
        }

        # 新增：模式名到模式编号的映射，方便测试指令使用
        self.mode_name_map = {
            'speed': '1', '2倍速': '1', 'speedup': '1',
            'reverse': '2', '倒放': '2',
            'piano': '3', '钢琴': '3',
            'accompaniment': '4', '伴奏': '4',
            'bass': '5', '贝斯': '5',
            'drums': '6', '鼓组': '6',
            'vocals': '7', '人声': '7',
        }

        # --- 新增：创建受控的线程池 ---
        # 对于CPU密集型任务，将工作线程数限制为1，以避免在低核心CPU上发生线程争抢。
        # 所有耗时的音频处理都将在这个队列中排队执行。
        self.executor = ThreadPoolExecutor(max_workers=1)

        # --- 新增：初始化后台任务句柄 ---
        self._cleanup_task = None

        # --- 新增：为所有歌曲版本创建 bundle_name -> song 的映射，提高查找效率 ---
        if self.songs_data:
            for song in self.songs_data:
                for v in song.get('vocals', []):
                    self.bundle_to_song_map[v['vocalAssetbundleName']] = song
        
        # --- 改造：根据配置扫描本地或拉取远程 manifest 来加载可用音轨 ---
        self.preprocessed_tracks = {
            "accompaniment": set(), "bass_only": set(),
            "drums_only": set(), "vocals_only": set()
        }
        self.available_piano_songs_bundles = set()

        use_local = self.config.get("use_local_resources", True)

        if use_local:
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
        else:
            logger.info("使用远程资源模式，开始获取 manifest.json...")
            manifest_url = self._get_resource_path_or_url("manifest.json")
            if manifest_url and isinstance(manifest_url, str):
                try:
                    with urlopen(manifest_url, timeout=10) as response:
                        manifest_data = json.load(response)
                        
                        for mode in self.preprocessed_tracks.keys():
                            self.preprocessed_tracks[mode] = set(manifest_data.get(mode, []))
                            logger.info(f"成功从 manifest 加载 {len(self.preprocessed_tracks[mode])} 个 '{mode}' 模式的音轨。")
                        
                        self.available_piano_songs_bundles = set(manifest_data.get("songs_piano_trimmed_mp3", []))
                        logger.info(f"成功从 manifest 加载 {len(self.available_piano_songs_bundles)} 个钢琴模式的音轨。")

                except (URLError, json.JSONDecodeError, Exception) as e:
                    logger.error(f"获取或解析远程 manifest.json 失败: {e}。插件将无法使用预处理音轨模式。", exc_info=True)
            else:
                 logger.error("无法构建 manifest.json 的 URL。插件将无法使用预处理音轨模式。")

        # --- 根据加载的音轨信息，填充可用的歌曲列表 ---
        if self.songs_data:
            song_list_map = {
                'accompaniment': (self.available_accompaniment_songs, set()),
                'vocals_only': (self.available_vocals_songs, set()),
                'bass_only': (self.available_bass_songs, set()),
                'drums_only': (self.available_drums_songs, set()),
            }

            for mode, bundles in self.preprocessed_tracks.items():
                # 注意：这里需要检查 key 是否存在
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

        # 筛选出有 another_vocal 的歌曲
        self.another_vocal_songs = []
        if self.songs_data:
            for song in self.songs_data:
                if any(v.get('musicVocalType') == 'another_vocal' for v in song.get('vocals', [])):
                    self.another_vocal_songs.append(song)
        
        # 使用 context 初始化共享的游戏会话状态
        if not hasattr(self.context, "active_game_sessions"):
            self.context.active_game_sessions = set()

        if not self.songs_data:
            logger.error("插件初始化失败，缺少歌曲数据文件。")
        if not PYDUB_AVAILABLE:
            logger.error("音频处理库 'pydub' 未找到。猜歌功能将无法使用。请运行 'pip install pydub' 并确保已安装 'ffmpeg'。")
            
        os.makedirs(self.output_dir, exist_ok=True)
        # 启动时清理一次
        self._cleanup_output_dir()
        # --- 新增：启动周期性清理任务 ---
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup_task())

    async def _periodic_cleanup_task(self):
        """每隔一小时自动清理一次 output 目录。"""
        cleanup_interval_seconds = 3600 # 1 hour
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

    def _open_image(self, relative_path: str) -> Optional[Image.Image]:
        """打开一个资源图片，无论是本地路径还是远程URL。"""
        source = self._get_resource_path_or_url(relative_path)
        if not source:
            return None
        
        try:
            # 如果 source 是字符串且以 http 开头，则视为URL
            if isinstance(source, str) and source.startswith(('http://', 'https://')):
                with urlopen(source) as response:
                    img = Image.open(response)
                    # 必须在 with 块结束前加载图像数据，否则文件流会关闭
                    img.load() 
                    return img
            else: # 否则，视为本地 Path 对象
                return Image.open(source)
        except (URLError, Exception) as e:
            logger.error(f"无法打开图片资源 {source}: {e}", exc_info=True)
            return None

    def _get_duration_ms_ffprobe(self, file_path: Union[Path, str]) -> Optional[float]:
        """使用 ffprobe 高效获取音频时长，避免用 pydub 加载整个文件。"""
        command = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(file_path)
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
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

    def _create_options_image(self, options: List[Dict]) -> Optional[str]:
        """为12个歌曲选项创建一个3x4的图鉴"""
        if not options or len(options) != 12:
            return None

        jacket_w, jacket_h = 128, 128
        padding = 15
        text_h = 50 
        cols, rows = 3, 4
        
        img_w = cols * jacket_w + (cols + 1) * padding
        img_h = rows * (jacket_h + text_h) + (rows + 1) * padding

        img = Image.new('RGBA', (img_w, img_h), (245, 245, 245, 255))
        
        try:
            # 确保使用新字体
            font_path = str(self.resources_dir / "font.ttf")
            title_font = ImageFont.truetype(font_path, 16)
            # 增大标号字体
            num_font = ImageFont.truetype(font_path, 22) 
        except IOError:
            title_font = ImageFont.load_default()
            num_font = title_font

        draw = ImageDraw.Draw(img)

        for i, option in enumerate(options):
            row_idx, col_idx = i // cols, i % cols
            
            x = padding + col_idx * (jacket_w + padding)
            y = padding + row_idx * (jacket_h + text_h + padding)

            try:
                jacket_path = self.resources_dir / "music_jacket" / f"{option['jacketAssetbundleName']}.png"
                if not jacket_path.exists(): continue

                jacket = Image.open(jacket_path).convert("RGBA").resize((jacket_w, jacket_h), LANCZOS)
                img.paste(jacket, (x, y), jacket)
                
                # --- 优化标号显示 ---
                num_text = f"{i + 1}"
                
                # 绘制一个圆形的背景
                circle_radius = 16
                circle_center = (x + circle_radius, y + circle_radius)
                draw.ellipse(
                    (circle_center[0] - circle_radius, circle_center[1] - circle_radius,
                     circle_center[0] + circle_radius, circle_center[1] + circle_radius),
                    fill=(0, 0, 0, 180) # 半透明黑色背景
                )
                
                # 在圆形中心绘制文本
                pilmoji_drawer = Pilmoji(img)
                pilmoji_drawer.text(circle_center, num_text, font=num_font, fill=(255, 255, 255), anchor="mm")

                # 绘制标题
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

    def start_new_game(self, **kwargs) -> Optional[Dict]:
        """
        准备一轮新游戏。
        该函数现在会智能选择处理路径：
        - 快速路径：对简单裁剪任务直接使用ffmpeg，性能更高。
        - 慢速路径：对需要变速、倒放等复杂效果的任务，使用pydub。
        """
        if not self.songs_data or not PYDUB_AVAILABLE:
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
                song = random.choice(self.songs_data)
        
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
        # 路径A: 快速路径 (直接使用ffmpeg，性能高)
        if not use_slow_path:
            try:
                total_duration_ms = self._get_duration_ms_ffprobe(audio_source)
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
                
                result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
                if result.returncode != 0:
                    raise RuntimeError(f"ffmpeg clipping failed: {result.stderr}")
                
                mode = "normal"
                if preprocessed_mode: mode = preprocessed_mode
                elif is_piano_mode: mode = "melody_to_piano"
                
                return {"song": song, "clip_path": str(clip_path_obj), "score": kwargs.get("score", 1), "mode": mode}

            except Exception as e:
                logger.warning(f"快速路径处理失败: {e}. 将回退到 pydub 慢速路径。")
        
        # 路径B: 慢速路径 (使用pydub，兼容复杂效果)
        try:
            # --- 修正：为 pydub 处理远程URL ---
            if isinstance(audio_source, str) and audio_source.startswith(('http://', 'https://')):
                # pydub不能直接打开URL，所以我们先下载到内存中
                with urlopen(audio_source) as response:
                    buffer = io.BytesIO(response.read())
                audio = AudioSegment.from_file(buffer, format=audio_format)
            else:
                # 本地文件路径，pydub可以直接处理
                audio = AudioSegment.from_file(audio_source, format=audio_format)

            if preprocessed_mode == "bass_only":
                audio += 6

            target_duration_ms = int(self.config.get("clip_duration_seconds", 10) * 1000)
            if preprocessed_mode in ["bass_only", "drums_only"]:
                target_duration_ms *= 2
            
            speed_multiplier = kwargs.get("speed_multiplier", 1.0)
            source_duration_ms = int(target_duration_ms * speed_multiplier)
            total_duration_ms = len(audio)
            
            if source_duration_ms >= total_duration_ms:
                clip_segment = audio
            else:
                start_range_min = 0
                if not preprocessed_mode and not is_piano_mode:
                    start_range_min = int(song.get("fillerSec", 0) * 1000)
                
                start_range_max = total_duration_ms - source_duration_ms
                start_ms = random.randint(start_range_min, start_range_max) if start_range_min < start_range_max else start_range_min
                end_ms = start_ms + source_duration_ms
                clip_segment = audio[start_ms:end_ms]

            clip = clip_segment
            
            if speed_multiplier != 1.0:
                clip = clip._spawn(clip.raw_data, overrides={'frame_rate': int(clip.frame_rate * speed_multiplier)})
            if kwargs.get("reverse_audio", False):
                clip = clip.reverse()
            
            if has_band_pass and isinstance(has_band_pass, tuple) and len(has_band_pass) == 2:
                low_freq, high_freq = has_band_pass
                clip = clip.high_pass_filter(low_freq).low_pass_filter(high_freq) + 6

            mode = "normal"
            if preprocessed_mode: mode = preprocessed_mode
            elif is_piano_mode: mode = "melody_to_piano"
            elif has_band_pass: mode = "band_pass"
            elif has_reverse: mode = "reverse"
            elif has_speed_change: mode = "speedup"
            
            if kwargs.get("random_mode_name"):
                mode = kwargs["random_mode_name"]

            clip_path = self.output_dir / f"clip_{int(time.time())}.mp3"
            clip.export(clip_path, format="mp3", bitrate="128k")

            return {"song": song, "clip_path": str(clip_path), "score": kwargs.get("score", 1), "mode": mode}

        except Exception as e:
            logger.error(f"慢速路径 (pydub) 处理音频文件 {audio_source} 时失败: {e}", exc_info=True)
            return None

    def _check_game_start_conditions(self, event: AstrMessageEvent) -> Tuple[bool, Optional[str]]:
        """检查是否可以开始新游戏，返回(布尔值, 提示信息)"""
        if not self._is_group_allowed(event): 
            return False, None
            
        session_id = event.unified_msg_origin
        cooldown = self.config.get("game_cooldown_seconds", 60)
        debug_mode = self.config.get("debug_mode", False)
        
        if not debug_mode and time.time() - self.last_game_end_time.get(session_id, 0) < cooldown:
            remaining_time = int(cooldown - (time.time() - self.last_game_end_time.get(session_id, 0)))
            return False, f"嗯......休息 {remaining_time} 秒再玩吧......"
        
        if session_id in self.context.active_game_sessions:
            return False, "......有一个正在进行的游戏了呢。"
        
        if not debug_mode and not self._can_play(event.get_sender_id()):
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

            if game_data.get("game_mode", "song") == "song":
                self._update_stats(user_id, user_name, score_to_add, correct=is_correct)
                self._update_mode_stats(game_data.get('mode', 'normal'), is_correct)

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
        
        summary_prefix = f"本轮猜测已达上限({max_guess_attempts}次)！" if game_ended_by_attempts else "时间到！"
        if correct_players:
            winner_names = "、".join(player['name'] for player in correct_players.values())
            summary_text = f"{summary_prefix}\n本轮答对的玩家有：\n{winner_names}"
            await event.send(event.plain_result(summary_text))
        else:
            summary_text = f"{summary_prefix} 好像......没有人答对......"
            await event.send(event.plain_result(summary_text))
            
        await event.send(event.chain_result(answer_reveal_messages))

    async def _start_game_logic(self, event: AstrMessageEvent, **kwargs):
        """猜歌游戏核心逻辑(准备阶段)"""
        can_start, message = self._check_game_start_conditions(event)
        if not can_start:
            if message:
                await event.send(event.plain_result(message))
            return

        session_id = event.unified_msg_origin
        debug_mode = self.config.get("debug_mode", False)

        # 标记会话并记录次数
        self.context.active_game_sessions.add(session_id)
        # --- 新增：发送统计信标 ---
        asyncio.create_task(self._send_stats_ping("guess_song"))
        if not debug_mode:
            self._record_game_start(event.get_sender_id(), event.get_sender_name())
        
        # 将同步的音频处理任务扔到线程池中执行
        try:
            loop = asyncio.get_running_loop()
            game_data_callable = partial(self.start_new_game, **kwargs)
            game_data = await loop.run_in_executor(
                self.executor, game_data_callable
            )
        except Exception as e:
            logger.error(f"在线程池中执行 start_new_game 失败: {e}", exc_info=True)
            game_data = None

        if not game_data:
            await event.send(event.plain_result("......开始游戏失败，可能是缺少资源文件、配置错误或ffmpeg未安装，请联系管理员。"))
            if session_id in self.context.active_game_sessions:
                self.context.active_game_sessions.remove(session_id)
            return
        
        if not self.songs_data:
            await event.send(event.plain_result("......歌曲数据未加载，无法生成选项。"))
            if session_id in self.context.active_game_sessions:
                self.context.active_game_sessions.remove(session_id)
            return

        # 准备选项
        correct_song = game_data['song']
        other_songs = random.sample([s for s in self.songs_data if s['id'] != correct_song['id']], 11)
        options = [correct_song] + other_songs
        random.shuffle(options)
        
        # 更新游戏数据
        game_data['options'] = options
        game_data['correct_answer_num'] = options.index(correct_song) + 1
        game_data['game_mode'] = 'song'
        
        logger.info(f"[猜歌插件] 新游戏开始. 答案: {correct_song['title']} (选项 {game_data['correct_answer_num']})")
        
        # 准备消息
        options_img_path = self._create_options_image(options)
        timeout_seconds = self.config.get("answer_timeout", 30)
        intro_text = f".......嗯\n这首歌是？请在{timeout_seconds}秒内发送编号回答。\n"
        
        intro_messages = [Comp.Plain(intro_text)]
        if options_img_path:
            intro_messages.append(Comp.Image(file=options_img_path))
            
            jacket_path = self.resources_dir / "music_jacket" / f"{correct_song['jacketAssetbundleName']}.png"
        answer_reveal_messages = [
                Comp.Plain(f"正确答案是: {game_data['correct_answer_num']}. {correct_song['title']}\n"),
                Comp.Image(file=str(jacket_path))
            ]

        # 启动游戏会话
        await self._run_game_session(event, game_data, intro_messages, answer_reveal_messages)

    @filter.command(
        "猜歌",
        alias={
            "gs",
            # 添加所有带数字的指令作为别名，以确保它们能被路由到这个统一的处理器
            "猜歌1", "猜歌2", "猜歌3", "猜歌4", "猜歌5", "猜歌6", "猜歌7",
            "gs1", "gs2", "gs3", "gs4", "gs5", "gs6", "gs7"
        }
    )
    async def start_guess_song_unified(self, event: AstrMessageEvent):
        """统一处理所有固定模式的猜歌指令"""
        # 提取模式编号
        match = re.search(r'\d+', event.message_str)
        mode_num_str = match.group(0) if match else None

        # --- 新增：轻量模式检查 ---
        is_lightweight = self.config.get("lightweight_mode", False)
        if is_lightweight and mode_num_str in ['1', '2']: # 1=2倍速, 2=倒放
            await event.send(event.plain_result("......轻量模式已启用，此模式不可用。"))
            return
        # --- 结束 ---

        # 从映射中获取游戏参数
        game_kwargs = self.game_modes.get(mode_num_str, {'kwargs': {'score': 1}})['kwargs']
        
        await self._start_game_logic(event, **game_kwargs)

    @filter.command("随机猜歌", alias={"rgs"})
    async def start_random_guess_song(self, event: AstrMessageEvent):
        """开始一轮随机特殊模式的猜歌，可能叠加多种效果"""
        can_start, message = self._check_game_start_conditions(event)
        if not can_start:
            if message:
                await event.send(event.plain_result(message))
            return
            
        session_id = event.unified_msg_origin
        debug_mode = self.config.get("debug_mode", False)

        # 标记会话并记录次数
        self.context.active_game_sessions.add(session_id)
        # --- 新增：为随机模式发送统计信标 ---
        asyncio.create_task(self._send_stats_ping("guess_song_random"))
        if not debug_mode:
            self._record_game_start(event.get_sender_id(), event.get_sender_name())
        
        # --- 新逻辑：根据分数加权随机选择效果组合 ---
        # --- 新增：轻量模式处理 ---
        is_lightweight = self.config.get("lightweight_mode", False)
        
        # 基础效果（改变音频特性）
        base_effects = [
            {'name': '2倍速', 'kwargs': {'speed_multiplier': 2.0}, 'score': 1, 'group': 'speed'},
            {'name': '倒放', 'kwargs': {'reverse_audio': True}, 'score': 3, 'group': 'direction'},
        ]

        if is_lightweight:
            base_effects = []
        # --- 结束 ---

        # 音源效果（选择播放内容，互相排斥）
        source_effects = [
            # {'name': '带通滤波', 'kwargs': {'band_pass': (1, 200)}, 'score': 1, 'group': 'source'},
            {'name': 'Twin Piano ver.', 'kwargs': {'melody_to_piano': True}, 'score': 2, 'group': 'source'},
            {'name': '纯伴奏', 'kwargs': {'play_preprocessed': 'accompaniment'}, 'score': 1, 'group': 'source'},
            {'name': '纯贝斯', 'kwargs': {'play_preprocessed': 'bass_only'}, 'score': 3, 'group': 'source'},
            {'name': '纯鼓组', 'kwargs': {'play_preprocessed': 'drums_only'}, 'score': 4, 'group': 'source'},
            {'name': '纯人声', 'kwargs': {'play_preprocessed': 'vocals_only'}, 'score': 1, 'group': 'source'},
        ]

        all_combinations = []
        weights = []

        # 生成所有可能的组合 (最多1个音源效果 + 最多2个基础效果)
        for i in range(1, len(base_effects) + 2): # i 是组合中的效果总数
            # 组合一：只包含一个音源效果
            if i == 1:
                for effect in source_effects:
                    all_combinations.append([effect])
            
            # 组合二：一个音源效果 + 1到多个基础效果
            for source_effect in source_effects:
                # 从基础效果中选出 i-1 个
                if i > 1 and i - 1 <= len(base_effects):
                    for base_combo in itertools.combinations(base_effects, i - 1):
                        all_combinations.append([source_effect] + list(base_combo))

        # --- 重新计算权重和最终参数 ---
        final_combinations = []
        for combo in all_combinations:
            current_kwargs = {}
            current_score = 0
            current_names = []
            has_reverse = False
            has_speed_up = False
            speed_multiplier_value = 1.0

            is_speedup_chosen = any('2倍速' in e['name'] for e in combo)
            is_multi_effect_speedup = is_speedup_chosen and len(combo) > 1

            temp_combo = list(combo)
            if is_multi_effect_speedup:
                for idx, effect in enumerate(temp_combo):
                    if '2倍速' in effect['name']:
                        modified_effect = effect.copy()
                        modified_effect['kwargs'] = effect['kwargs'].copy()
                        modified_effect['kwargs']['speed_multiplier'] = 1.5
                        modified_effect['name'] = '1.5倍速'
                        temp_combo[idx] = modified_effect
                        break

            for effect in temp_combo:
                current_kwargs.update(effect['kwargs'])
                current_score += effect['score']
                current_names.append(effect['name'])
                if 'reverse_audio' in effect['kwargs']:
                    has_reverse = True
                if 'speed_multiplier' in effect['kwargs']:
                    has_speed_up = True
                    speed_multiplier_value = effect['kwargs']['speed_multiplier']
            
            if has_reverse and has_speed_up:
                current_score += 1

            # 分数越高，权重越低
            weight = 1 / (current_score ** 2) if current_score > 0 else 1 
            final_combinations.append({
                'kwargs': current_kwargs,
                'score': current_score,
                'names': current_names,
                'has_reverse': has_reverse,
                'has_speed_up': has_speed_up,
                'speed_multiplier_value': speed_multiplier_value,
                'weight': weight
            })

        # 根据权重随机选择一个组合
        chosen_combination = random.choices(
            population=final_combinations,
            weights=[c['weight'] for c in final_combinations],
            k=1
        )[0]

        # --- 从选中的组合中提取最终参数 ---
        combined_kwargs = chosen_combination['kwargs']
        total_score = chosen_combination['score']
        effect_names_display = sorted(chosen_combination['names'])

        # 修正组合效果的额外加分显示
        if chosen_combination['has_reverse'] and chosen_combination['has_speed_up']:
             # 找到并移除 "倒放" 和 "X倍速"，替换为组合名
            effect_names_display = [n for n in effect_names_display if n not in ['倒放', '2倍速', '1.5倍速']]
            effect_names_display.append(f"倒放+{chosen_combination['speed_multiplier_value']}倍速组合(+1分)")

        combined_kwargs['score'] = total_score
        # 记录本轮随机题型组合名
        combined_kwargs['random_mode_name'] = 'random_' + '+'.join(sorted(chosen_combination['names']))
        
        # 显示将要应用的效果
        effects_text = "、".join(effect_names_display)
        await event.send(event.plain_result(f"......随机模式启动！本轮应用效果：【{effects_text}】(总计{total_score}分)"))
        
        # 准备游戏数据
        try:
            loop = asyncio.get_running_loop()
            game_data_callable = partial(self.start_new_game, **combined_kwargs)
            game_data = await loop.run_in_executor(
                self.executor, game_data_callable
            )
        except Exception as e:
            logger.error(f"在线程池中执行 start_new_game 失败: {e}", exc_info=True)
            game_data = None
            
        if not game_data:
            await event.send(event.plain_result("......开始游戏失败，可能是缺少资源文件、配置错误或ffmpeg未安装，请联系管理员。"))
            if session_id in self.context.active_game_sessions:
                self.context.active_game_sessions.remove(session_id)
            return
            
        # 继续游戏逻辑
        if not self.songs_data:
            await event.send(event.plain_result("......歌曲数据未加载，无法生成选项。"))
            if session_id in self.context.active_game_sessions:
                self.context.active_game_sessions.remove(session_id)
            return

        correct_song = game_data['song']
        other_songs = random.sample([s for s in self.songs_data if s['id'] != correct_song['id']], 11)
        options = [correct_song] + other_songs
        random.shuffle(options)
        
        game_data['options'] = options
        game_data['correct_answer_num'] = options.index(correct_song) + 1
        game_data['game_mode'] = 'song'
        
        logger.info(f"[猜歌插件] 新游戏开始. 答案: {correct_song['title']} (选项 {game_data['correct_answer_num']})")
        
        options_img_path = self._create_options_image(options)
        timeout_seconds = self.config.get("answer_timeout", 30)
        intro_text = f".......嗯\n这首歌是？请在{timeout_seconds}秒内发送编号回答。\n"
        
        intro_messages = [Comp.Plain(intro_text)]
        if options_img_path:
            intro_messages.append(Comp.Image(file=options_img_path))
        
        jacket_path = self.resources_dir / "music_jacket" / f"{correct_song['jacketAssetbundleName']}.png"
        answer_reveal_messages = [
            Comp.Plain(f"正确答案是: {game_data['correct_answer_num']}. {correct_song['title']}\n"),
            Comp.Image(file=str(jacket_path))
        ]
        
        await self._run_game_session(event, game_data, intro_messages, answer_reveal_messages)

    @filter.command("猜歌手")
    async def start_vocalist_game(self, event: AstrMessageEvent):
        """开始一轮 '猜歌手' 游戏"""
        can_start, message = self._check_game_start_conditions(event)
        if not can_start:
            if message:
                await event.send(event.plain_result(message))
            return
        
        if not self.another_vocal_songs:
            await event.send(event.plain_result("......抱歉，没有找到包含 another_vocal 的歌曲，无法开始游戏。"))
            return
        
        session_id = event.unified_msg_origin
        debug_mode = self.config.get("debug_mode", False)

        # 标记会话并记录次数
        self.context.active_game_sessions.add(session_id)
        # --- 新增：为猜歌手模式发送统计信标 ---
        asyncio.create_task(self._send_stats_ping("guess_song_vocalist"))
        if not debug_mode:
            self._record_game_start(event.get_sender_id(), event.get_sender_name())
        
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
            loop = asyncio.get_running_loop()
            game_data_callable = partial(
                self.start_new_game,
                force_song_object=song,
                force_vocal_version=correct_vocal_version,
                speed_multiplier=1.5
            )
            game_data = await loop.run_in_executor(self.executor, game_data_callable)
        except Exception as e:
            logger.error(f"在线程池中执行 start_new_game 失败: {e}", exc_info=True)
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
        game_data['game_mode'] = 'vocalist' # 标记为猜歌手模式

        def get_vocalist_name(vocal_info):
            char_ids = [c['characterId'] for c in vocal_info.get('characters', [])]
            names = [self.character_map.get(str(cid), "未知角色") for cid in char_ids]
            return " & ".join(names)

        compact_options_text = ""
        for i, vocal in enumerate(another_vocals):
            vocalist_name = get_vocalist_name(vocal)
            compact_options_text += f"{i + 1}. {vocalist_name}\n"
        
        # 3. 准备消息
        timeout_seconds = self.config.get("answer_timeout", 30)
        intro_text = f"这首歌是【{song['title']}】，正在演唱的是谁？[1.5倍速]\n请在{timeout_seconds}秒内发送编号回答。\n\n⚠️ 测试功能，不计分\n\n{compact_options_text}"
        jacket_path = self.resources_dir / "music_jacket" / f"{song['jacketAssetbundleName']}.png"
        
        intro_messages = [Comp.Plain(intro_text)]
        if jacket_path.exists():
            intro_messages.append(Comp.Image(file=str(jacket_path)))
            
        correct_vocalist_name = get_vocalist_name(correct_vocal_version)
        answer_reveal_messages = [
            Comp.Plain(f"正确答案是: {game_data['correct_answer_num']}. {correct_vocalist_name}")
        ]

        # 4. 启动游戏
        await self._run_game_session(event, game_data, intro_messages, answer_reveal_messages)

    @filter.command("猜歌帮助")
    async def show_guess_song_help(self, event: AstrMessageEvent):
        """显示猜歌插件帮助"""
        if not self._is_group_allowed(event):
            return
        help_text = (
            "--- 猜歌插件帮助 ---\n\n"
            "🎵 基础指令\n"
            "  猜歌 - 普通模式 (1分)\n"
            "  猜歌 1 - 2倍速 (1分)\n"
            "  猜歌 2 - 倒放 (3分)\n"
            "  猜歌 3 - AI-Assisted Twin Piano ver. (2分)\n"
            "  猜歌 4 - 纯伴奏模式 (1分)\n"
            "  猜歌 5 - 纯贝斯模式 (3分)\n"
            "  猜歌 6 - 纯鼓组模式 (4分)\n"
            "  猜歌 7 - 纯人声模式 (1分)\n\n"
            "🎲 高级指令\n"
            "  随机猜歌 - 随机组合效果 (最高9分)\n"
            "  猜歌手 - 竞猜演唱者 (测试功能, 不计分)\n"
            "  听<模式> [歌曲名/ID] - 播放指定或随机歌曲的特殊音轨。\n"
            "    可用模式: 钢琴, 伴奏, 人声, 贝斯, 鼓组\n"
            "    (该功能有统一的每日次数限制)\n\n"
            "📊 数据统计\n"
            "  猜歌分数 - 查看自己的猜歌积分和排名\n"
            "  查看统计 - 查看各题型的正确率排行"
        )
        await event.send(event.plain_result(help_text))

    @filter.command("猜歌排行榜", alias={"gssrank", "gstop"})
    async def show_ranking(self, event: AstrMessageEvent):
        """显示猜歌排行榜"""
        if not self._is_group_allowed(event): return
        self._cleanup_output_dir()

        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id, user_name, score, attempts, correct_attempts FROM user_stats ORDER BY score DESC LIMIT 10")
            rows = cursor.fetchall()

        if not rows:
            yield event.plain_result("......目前还没有人参与过猜歌游戏")
            return

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
                title_text = "猜歌排行榜"
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
                    pilmoji.text((col_positions[4], current_y), attempts, font=body_font, fill=font_color)

                    if i < len(rows) - 1:
                        draw = ImageDraw.Draw(img)
                        draw.line([(30, current_y + 60), (width - 30, current_y + 60)], fill=(200, 200, 210, 128), width=1)
                    
                    current_y += 70

                footer_text = f"GuessSong v{PLUGIN_VERSION} | Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                pilmoji.text((center_x, height - 25), footer_text, font=id_font, fill=header_color, anchor="ms")

            img_path = self.output_dir / f"song_ranking_{int(time.time())}.png"
            img.save(img_path)
            yield event.image_result(str(img_path))

        except Exception as e:
            logger.error(f"生成猜歌排行榜失败: {e}", exc_info=True)
            yield event.plain_result("生成排行榜图片时出错。")
            
    @filter.command("猜歌分数", alias={"gsscore", "我的猜歌分数"})
    async def show_user_score(self, event: AstrMessageEvent):
        """显示玩家自己的猜歌积分和统计数据"""
        if not self._is_group_allowed(event): return
        user_id = event.get_sender_id()
        user_name = event.get_sender_name()
        
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT score, attempts, correct_attempts, last_play_date, daily_plays FROM user_stats WHERE user_id = ?", (user_id,))
            user_data = cursor.fetchone()
            
        if not user_data:
            yield event.plain_result(f"......{user_name}，你还没有参与过猜歌游戏哦。")
            return
            
        score, attempts, correct_attempts, last_play_date, daily_plays = user_data
        accuracy = (correct_attempts * 100 / attempts) if attempts > 0 else 0
        
        # 计算排名
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM user_stats WHERE score > ?", (score,))
            rank = cursor.fetchone()[0] + 1  # 排名 = 比自己分数高的人数 + 1
        
        daily_limit = self.config.get("daily_play_limit", 15)
        remaining_plays = daily_limit - daily_plays if last_play_date == time.strftime("%Y-%m-%d") else daily_limit
        
        stats_text = (
            f"--- {user_name} 的猜歌数据 ---\n"
            f"🏆 总分: {score} 分\n"
            f"🎯 正确率: {accuracy:.1f}%\n"
            f"🎮 游戏次数: {attempts} 次\n"
            f"✅ 答对次数: {correct_attempts} 次\n"
            f"🏅 当前排名: 第 {rank} 名\n"
            f"📅 今日剩余游戏次数: {remaining_plays} 次"
        )
        
        yield event.plain_result(stats_text)
    
    # --- Data and state management methods ---
    def _record_game_start(self, user_id: str, user_name: str):
        with self.get_conn() as conn:
            cursor, today = conn.cursor(), time.strftime("%Y-%m-%d")
            cursor.execute("SELECT last_play_date, daily_plays FROM user_stats WHERE user_id = ?", (user_id,))
            user_data = cursor.fetchone()
            if user_data:
                last_play_date, _ = user_data
                if last_play_date == today:
                    cursor.execute("UPDATE user_stats SET user_name = ?, daily_plays = daily_plays + 1 WHERE user_id = ?", (user_name, user_id))
                else:
                    cursor.execute("UPDATE user_stats SET user_name = ?, last_play_date = ?, daily_plays = 1, daily_listen_plays = 0 WHERE user_id = ?", (user_name, today, user_id))
            else:
                cursor.execute("INSERT INTO user_stats (user_id, user_name, last_play_date, daily_plays, daily_listen_plays) VALUES (?, ?, ?, 1, 0)", (user_id, user_name, today))
            conn.commit()

    def _record_listen_song(self, user_id: str, user_name: str):
        """统一记录听歌（钢琴/伴奏）次数"""
        with self.get_conn() as conn:
            cursor, today = conn.cursor(), time.strftime("%Y-%m-%d")
            cursor.execute("SELECT last_play_date, daily_listen_plays FROM user_stats WHERE user_id = ?", (user_id,))
            user_data = cursor.fetchone()
            if user_data:
                last_play_date, _ = user_data
                if last_play_date == today:
                    cursor.execute("UPDATE user_stats SET user_name = ?, daily_listen_plays = daily_listen_plays + 1 WHERE user_id = ?", (user_name, user_id))
                else:
                    cursor.execute("UPDATE user_stats SET user_name = ?, last_play_date = ?, daily_plays = 0, daily_listen_plays = 1 WHERE user_id = ?", (user_name, today, user_id))
            else:
                cursor.execute("INSERT INTO user_stats (user_id, user_name, last_play_date, daily_plays, daily_listen_plays) VALUES (?, ?, ?, 0, 1)", (user_id, user_name, today))
            conn.commit()

    def _update_stats(self, user_id: str, user_name: str, score: int, correct: bool):
        """（已重构）使用原子化操作安全地更新用户统计数据。"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            # 尝试更新现有用户。SET 子句中的表达式是原子性的。
            cursor.execute(
                """
                UPDATE user_stats
                SET
                    user_name = ?,
                    score = score + ?,
                    attempts = attempts + 1,
                    correct_attempts = correct_attempts + ?
                WHERE user_id = ?
                """,
                (user_name, score, 1 if correct else 0, user_id)
            )

            # 如果没有行被更新，说明用户不存在，则插入新记录。
            if cursor.rowcount == 0:
                cursor.execute(
                    """
                    INSERT INTO user_stats
                        (user_id, user_name, score, attempts, correct_attempts, last_play_date, daily_plays, daily_listen_plays)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, user_name, score, 1, 1 if correct else 0, time.strftime("%Y-%m-%d"), 0, 0)
                )
            conn.commit()

    def _can_play(self, user_id: str) -> bool:
        daily_limit = self.config.get("daily_play_limit", 15)
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT daily_plays, last_play_date FROM user_stats WHERE user_id = ?", (user_id,))
            user_data = cursor.fetchone()
            return not (user_data and user_data[1] == time.strftime("%Y-%m-%d") and user_data[0] >= daily_limit)
            
    def _can_listen_song(self, user_id: str) -> bool:
        """统一检查听歌（钢琴/伴奏）次数"""
        daily_limit = self.config.get("daily_listen_limit", 5)
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT daily_listen_plays, last_play_date FROM user_stats WHERE user_id = ?", (user_id,))
            user_data = cursor.fetchone()
            return not (user_data and user_data[1] == time.strftime("%Y-%m-%d") and user_data[0] >= daily_limit)

    @filter.command("重置猜歌次数", alias={"resetgs"})
    async def reset_guess_limit(self, event: AstrMessageEvent):
        """重置用户猜歌次数（仅限管理员）"""
        if str(event.get_sender_id()) not in self.config.get("super_users", []):
            return
            
        parts = event.message_str.strip().split()
        target_id = parts[1] if len(parts) > 1 and parts[1].isdigit() else event.get_sender_id()

        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM user_stats WHERE user_id = ?", (str(target_id),))
            if cursor.fetchone():
                cursor.execute("UPDATE user_stats SET daily_plays = 0 WHERE user_id = ?", (str(target_id),))
                conn.commit()
                yield event.plain_result(f"......用户 {target_id} 的猜歌次数已重置。")
            else:
                yield event.plain_result(f"......未找到用户 {target_id} 的游戏记录。")

    @filter.command("重置听歌次数", alias={"resetls"})
    async def reset_listen_limit(self, event: AstrMessageEvent):
        """重置用户每日听歌次数（钢琴和伴奏）"""
        if str(event.get_sender_id()) not in self.config.get("super_users", []):
            return
            
        parts = event.message_str.strip().split()
        target_id = parts[1] if len(parts) > 1 and parts[1].isdigit() else event.get_sender_id()

        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM user_stats WHERE user_id = ?", (str(target_id),))
            if cursor.fetchone():
                cursor.execute("UPDATE user_stats SET daily_listen_plays = 0 WHERE user_id = ?", (str(target_id),))
                conn.commit()
                await event.send(event.plain_result(f"......用户 {target_id} 的听歌次数已重置。"))
            else:
                await event.send(event.plain_result(f"......未找到用户 {target_id} 的游戏记录。"))

    @filter.command("重置题型统计", alias={"resetmodestats"})
    async def reset_mode_stats(self, event: AstrMessageEvent):
        """清空所有题型统计数据（仅限管理员）"""
        if str(event.get_sender_id()) not in self.config.get("super_users", []):
            return
        
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM mode_stats")
            conn.commit()
        
        await event.send(event.plain_result("......所有题型统计数据已被清空。"))

    async def terminate(self):
        # --- 新增：在插件终止时关闭线程池和后台任务 ---
        logger.info("正在关闭猜歌插件的线程池和后台任务...")
        if self._cleanup_task:
            self._cleanup_task.cancel()
        self.executor.shutdown(wait=False)
        logger.info("猜歌插件已终止。")
        pass

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
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT mode, total_attempts, correct_attempts FROM mode_stats")
            rows = cursor.fetchall()
        if not rows:
            yield event.plain_result("暂无题型统计数据。"); return
        # 计算正确率并排序
        stats = []
        for mode, total, correct in rows:
            acc = (correct * 100 / total) if total > 0 else 0
            stats.append((mode, total, correct, acc))
        stats.sort(key=lambda x: x[3])  # 按正确率升序
        # 生成图片
        img = self._draw_mode_stats_image(stats)
        img_path = self.output_dir / f"mode_stats_{int(time.time())}.png"
        img.save(img_path)
        yield event.image_result(str(img_path)) 

    def _draw_mode_stats_image(self, stats):
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
        return img

    def _mode_display_name(self, mode):
        # 题型名美化
        mode_map = {
            "speed": "2倍速", "reverse": "倒放", "piano": "钢琴",
            "karaoke": "纯伴奏", "bass": "纯贝斯", "drums": "纯鼓组",
            "vocals": "纯人声", "normal": "普通"
        }
        # 匹配随机组合模式
        if mode.startswith("random_"):
            parts = mode.replace("random_", "").split('+')
            # 查找并美化每个部分
            display_parts = [mode_map.get(p, p) for p in parts]
            return "随机-" + "+".join(display_parts)
        return mode_map.get(mode, mode)

    @filter.command("测试猜歌", alias={"test_song", "调试猜歌"})
    async def test_guess_song(self, event: AstrMessageEvent):
        """(管理员) 生成一个用于测试的猜歌游戏，可指定歌曲和多种模式。"""
        if str(event.get_sender_id()) not in self.config.get("super_users", []):
            return

        # 新用法: /测试猜歌 [模式1,模式2,...] <歌曲名或ID>
        # 例如: /测试猜歌 5,2 Tell Your World
        # 例如: /测试猜歌 bass,reverse 21
        
        parts = event.message_str.strip().split(maxsplit=1)
        if len(parts) < 2:
            await event.send(event.plain_result("用法: /测试猜歌 [模式,...] <歌曲名或ID>\n例如: /测试猜歌 5,2 Tell Your World"))
            return

        args_str = parts[1]
        
        # --- 智能解析模式和歌曲 ---
        arg_parts = args_str.split()
        potential_modes_str = arg_parts[0]
        
        temp_modes = re.split(r'[,，]', potential_modes_str)
        are_all_modes = True
        parsed_mode_keys = []

        for mode_str in temp_modes:
            mode_key = self.mode_name_map.get(mode_str.lower(), mode_str)
            if mode_key not in self.game_modes:
                are_all_modes = False
                break
            parsed_mode_keys.append(mode_key)
        
        if are_all_modes:
            mode_keys_input = list(dict.fromkeys(parsed_mode_keys)) # 去重并保持顺序
            song_query = " ".join(arg_parts[1:])
        else:
            mode_keys_input = []
            song_query = args_str

        if not song_query:
            await event.send(event.plain_result("请输入要测试的歌曲名称或ID。"))
            return

        # --- 构建游戏参数 ---
        final_kwargs = {}
        effect_names = []
        total_score = 0

        if not mode_keys_input:
            final_kwargs = self.game_modes[None]['kwargs'].copy()
            effect_names.append("普通")
            total_score = final_kwargs.get('score', 1)
        else:
            for mode_key in mode_keys_input:
                mode_data = self.game_modes.get(mode_key)
                if mode_data:
                    final_kwargs.update(mode_data['kwargs'])
                    display_name_found = False
                    for name, key in self.mode_name_map.items():
                        if key == mode_key and not name.isdigit():
                            effect_names.append(name.capitalize())
                            display_name_found = True
                            break
                    if not display_name_found:
                        effect_names.append(f"模式{mode_key}")
                    total_score += mode_data['kwargs'].get('score', 0)
        
        # --- 查找歌曲（已修正逻辑） ---
        target_song = None
        if song_query.isdigit():
            # 如果是数字，只按ID搜索
            target_song = next((s for s in self.songs_data if s['id'] == int(song_query)), None)
        else:
            # 否则，按标题搜索
            exact_match = next((s for s in self.songs_data if s['title'].lower() == song_query.lower()), None)
            if exact_match:
                target_song = exact_match
            else:
                found_songs = [s for s in self.songs_data if song_query.lower() in s['title'].lower()]
                if found_songs:
                    # 选择最接近的匹配（最短的标题）
                    target_song = min(found_songs, key=lambda s: len(s['title']))
        
        if not target_song:
            await event.send(event.plain_result(f'未在数据库中找到与 "{song_query}" 匹配的歌曲。'))
            return

        final_kwargs['force_song_object'] = target_song

        # --- 在线程池中执行游戏准备 ---
        try:
            loop = asyncio.get_running_loop()
            game_data_callable = partial(self.start_new_game, **final_kwargs)
            game_data = await loop.run_in_executor(self.executor, game_data_callable)
        except Exception as e:
            logger.error(f"测试模式下，在线程池中执行 start_new_game 失败: {e}", exc_info=True)
            game_data = None

        if not game_data:
            await event.send(event.plain_result("......生成测试游戏失败，请检查日志。"))
            return

        # --- 发送测试结果 ---
        correct_song = game_data['song']
        other_songs = random.sample([s for s in self.songs_data if s['id'] != correct_song['id']], 11)
        options = [correct_song] + other_songs
        random.shuffle(options)
        correct_answer_num = options.index(correct_song) + 1

        options_img_path = self._create_options_image(options)
        
        effects_text = "、".join(sorted(list(set(effect_names)))) or "普通"
        intro_text = f"--- 测试模式 ---\n歌曲: {correct_song['title']}\n效果: {effects_text} (理论分数: {total_score})\n"
        
        msg_chain = [Comp.Plain(intro_text)]
        if options_img_path:
            msg_chain.append(Comp.Image(file=options_img_path))
        
        await event.send(event.chain_result(msg_chain))
        await asyncio.sleep(0.5)
        await event.send(event.chain_result([Comp.Record(file=game_data["clip_path"])]))
        await asyncio.sleep(0.5)

        jacket_path = self.resources_dir / "music_jacket" / f"{correct_song['jacketAssetbundleName']}.png"
        answer_msg = [
            Comp.Plain(f"[测试模式] 正确答案是: {correct_answer_num}. {correct_song['title']}\n"),
            Comp.Image(file=str(jacket_path))
        ]
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
        cooldown = self.config.get("game_cooldown_seconds", 60)
        
        if time.time() - self.last_game_end_time.get(session_id, 0) < cooldown:
            yield event.plain_result(f"嗯......休息 {int(cooldown - (time.time() - self.last_game_end_time.get(session_id, 0)))} 秒再玩吧......")
            return
            
        if session_id in self.context.active_game_sessions:
            yield event.plain_result("......有一个正在进行的游戏或播放任务了呢。")
            return

        user_id = event.get_sender_id()
        user_name = event.get_sender_name()

        if not self._can_listen_song(user_id):
            limit = self.config.get('daily_listen_limit', 5)
            yield event.plain_result(f"......你今天听歌的次数已达上限（{limit}次），请明天再来吧......")
            return
        
        if not config['available_songs']:
            yield event.plain_result(config['not_found_msg'])
            return

        # --- 新增：为听歌模式发送统计信标 ---
        asyncio.create_task(self._send_stats_ping(f"listen_{mode}"))

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
        self.context.active_game_sessions.add(session_id)
        try:
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
            jacket_path = self.resources_dir / "music_jacket" / f"{song['jacketAssetbundleName']}.png"
            
            msg_chain = [Comp.Plain(f"歌曲:{song['id']}. {song['title']} {config['title_suffix']}\n")]
            if jacket_path.exists():
                msg_chain.append(Comp.Image(file=str(jacket_path)))
            
            yield event.chain_result(msg_chain)
            yield event.chain_result([Comp.Record(file=str(mp3_source))])

            self._record_listen_song(user_id, user_name)
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

    def _execute_ping_request(self, ping_url: str):
        """[Helper] Synchronous function to execute the ping request. Meant for ThreadPoolExecutor."""
        try:
            # urlopen is a blocking call, perfect for the executor.
            with urlopen(ping_url, timeout=2):
                pass  # We just need the request to be made.
        except Exception as e:
            # It's better to log this for debugging, even if we don't let it crash.
            logger.warning(f"Stats ping to {ping_url} failed: {e}")

    async def _send_stats_ping(self, game_type: str):
        """(已重构) 向专用统计服务器的5000端口发送GET请求。"""
        if self.config.get("use_local_resources", True):
            return

        resource_url_base = self.config.get("remote_resource_url_base", "")
        if not resource_url_base:
            return

        try:
            # 从资源URL中提取协议和主机名，然后强制使用5000端口
            parsed_url = urlparse(resource_url_base)
            stats_server_root = f"{parsed_url.scheme}://{parsed_url.hostname}:5000"
            
            # 构建最终的统计请求URL
            ping_url = f"{stats_server_root}/stats_ping/{game_type}.ping"

            # 在后台线程池中发送请求
            loop = asyncio.get_running_loop()
            loop.run_in_executor(self.executor, self._execute_ping_request, ping_url)
        except Exception as e:
            logger.error(f"无法调度统计ping任务: {e}")