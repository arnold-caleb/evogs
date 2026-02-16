#!/usr/bin/env python3
"""
Create an interactive HTML viewer for rendered Gaussian Splatting videos.
Features:
- Side-by-side comparison of multiple videos
- Timeline slider for frame-by-frame navigation
- Play/pause controls
- Speed adjustment
- Forward/backward playback
- Frame counter
"""

import argparse
import base64
import os
from pathlib import Path


def video_to_base64(video_path):
    """Convert video file to base64 for embedding in HTML."""
    with open(video_path, 'rb') as f:
        video_data = f.read()
    return base64.b64encode(video_data).decode('utf-8')


def create_video_viewer_html(video_paths, output_html, titles=None, fps=30):
    """
    Create an interactive HTML viewer for comparing videos.
    
    Args:
        video_paths: List of paths to video files
        output_html: Path to save the HTML file
        titles: Optional list of titles for each video
        fps: Frames per second of the videos
    """
    if titles is None:
        titles = [f"Video {i+1}" for i in range(len(video_paths))]
    
    # Determine video layout
    num_videos = len(video_paths)
    if num_videos == 1:
        grid_cols = 1
    elif num_videos == 2:
        grid_cols = 2
    elif num_videos <= 4:
        grid_cols = 2
    else:
        grid_cols = 3
    
    # Embed videos as base64 (for small videos) or use file paths
    video_sources = []
    for path in video_paths:
        file_size = os.path.getsize(path)
        if file_size < 50 * 1024 * 1024:  # Less than 50MB - embed
            video_base64 = video_to_base64(path)
            video_sources.append(f"data:video/mp4;base64,{video_base64}")
        else:  # Large file - use relative path
            rel_path = os.path.relpath(path, os.path.dirname(output_html))
            video_sources.append(rel_path)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gaussian Splatting Video Viewer</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .video-grid {{
            display: grid;
            grid-template-columns: repeat({grid_cols}, 1fr);
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }}
        
        .video-container {{
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .video-container:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.2);
        }}
        
        .video-title {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            font-weight: 600;
            font-size: 1.1em;
            text-align: center;
        }}
        
        .video-wrapper {{
            position: relative;
            width: 100%;
            padding-bottom: 56.25%; /* 16:9 aspect ratio */
            background: #000;
        }}
        
        video {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
        }}
        
        .controls {{
            background: white;
            padding: 30px;
            border-top: 2px solid #e9ecef;
        }}
        
        .timeline {{
            margin-bottom: 25px;
        }}
        
        .timeline-slider {{
            width: 100%;
            height: 8px;
            border-radius: 4px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            outline: none;
            opacity: 0.7;
            transition: opacity 0.2s;
            cursor: pointer;
        }}
        
        .timeline-slider:hover {{
            opacity: 1;
        }}
        
        .timeline-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #667eea;
            cursor: pointer;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        
        .timeline-slider::-moz-range-thumb {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #667eea;
            cursor: pointer;
            border: none;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        
        .timeline-info {{
            display: flex;
            justify-content: space-between;
            margin-top: 10px;
            font-size: 0.95em;
            color: #6c757d;
        }}
        
        .button-group {{
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }}
        
        button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            font-size: 1em;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
            box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
        }}
        
        button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
        }}
        
        button:active {{
            transform: translateY(0);
        }}
        
        button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }}
        
        .speed-control {{
            display: flex;
            align-items: center;
            gap: 15px;
            justify-content: center;
            margin-top: 20px;
        }}
        
        .speed-control label {{
            font-weight: 600;
            color: #495057;
        }}
        
        .speed-control select {{
            padding: 8px 16px;
            border-radius: 6px;
            border: 2px solid #667eea;
            font-size: 1em;
            cursor: pointer;
            background: white;
            color: #495057;
        }}
        
        .info-panel {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 20px;
            border-radius: 12px;
            margin-top: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .info-item {{
            text-align: center;
            padding: 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        
        .info-label {{
            font-size: 0.85em;
            color: #6c757d;
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .info-value {{
            font-size: 1.5em;
            font-weight: 700;
            color: #667eea;
        }}
        
        .shortcuts {{
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 12px;
            border-left: 4px solid #667eea;
        }}
        
        .shortcuts h3 {{
            margin-bottom: 15px;
            color: #495057;
        }}
        
        .shortcuts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 10px;
        }}
        
        .shortcut {{
            display: flex;
            justify-content: space-between;
            padding: 8px 12px;
            background: white;
            border-radius: 6px;
            font-size: 0.9em;
        }}
        
        .shortcut-key {{
            font-weight: 600;
            color: #667eea;
            font-family: 'Courier New', monospace;
        }}
        
        .shortcut-desc {{
            color: #6c757d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 Gaussian Splatting Video Viewer</h1>
            <p>Interactive comparison and frame-by-frame analysis</p>
        </div>
        
        <div class="video-grid">
"""
    
    # Add video containers
    for i, (src, title) in enumerate(zip(video_sources, titles)):
        html_content += f"""            <div class="video-container">
                <div class="video-title">{title}</div>
                <div class="video-wrapper">
                    <video id="video{i}" preload="auto">
                        <source src="{src}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                </div>
            </div>
"""
    
    html_content += """        </div>
        
        <div class="controls">
            <div class="timeline">
                <input type="range" id="timeline" class="timeline-slider" min="0" max="100" value="0" step="0.1">
                <div class="timeline-info">
                    <span id="currentTime">0.00s</span>
                    <span id="frameInfo">Frame: 0 / 0</span>
                    <span id="duration">0.00s</span>
                </div>
            </div>
            
            <div class="button-group">
                <button id="playPauseBtn" onclick="togglePlayPause()">▶ Play</button>
                <button onclick="stepBackward()">⏪ Step Back</button>
                <button onclick="stepForward()">Step Forward ⏩</button>
                <button onclick="reversePlayback()">🔄 Reverse</button>
                <button onclick="resetVideo()">⏮ Reset</button>
            </div>
            
            <div class="speed-control">
                <label for="speedSelect">Playback Speed:</label>
                <select id="speedSelect" onchange="changeSpeed()">
                    <option value="0.25">0.25×</option>
                    <option value="0.5">0.5×</option>
                    <option value="1" selected>1×</option>
                    <option value="1.5">1.5×</option>
                    <option value="2">2×</option>
                    <option value="3">3×</option>
                </select>
            </div>
            
            <div class="info-panel">
                <div class="info-item">
                    <div class="info-label">Videos Loaded</div>
                    <div class="info-value" id="videoCount">""" + str(num_videos) + """</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Current Frame</div>
                    <div class="info-value" id="currentFrame">0</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Total Frames</div>
                    <div class="info-value" id="totalFrames">0</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Direction</div>
                    <div class="info-value" id="direction">Forward ➡️</div>
                </div>
            </div>
            
            <div class="shortcuts">
                <h3>⌨️ Keyboard Shortcuts</h3>
                <div class="shortcuts-grid">
                    <div class="shortcut">
                        <span class="shortcut-key">Space</span>
                        <span class="shortcut-desc">Play / Pause</span>
                    </div>
                    <div class="shortcut">
                        <span class="shortcut-key">← →</span>
                        <span class="shortcut-desc">Step backward / forward</span>
                    </div>
                    <div class="shortcut">
                        <span class="shortcut-key">R</span>
                        <span class="shortcut-desc">Reverse playback</span>
                    </div>
                    <div class="shortcut">
                        <span class="shortcut-key">0</span>
                        <span class="shortcut-desc">Reset to start</span>
                    </div>
                    <div class="shortcut">
                        <span class="shortcut-key">+ / -</span>
                        <span class="shortcut-desc">Speed up / down</span>
                    </div>
                    <div class="shortcut">
                        <span class="shortcut-key">F</span>
                        <span class="shortcut-desc">Fullscreen</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const videos = [""" + ",".join([f"document.getElementById('video{i}')" for i in range(num_videos)]) + """];
        const timeline = document.getElementById('timeline');
        const playPauseBtn = document.getElementById('playPauseBtn');
        const currentTimeDisplay = document.getElementById('currentTime');
        const durationDisplay = document.getElementById('duration');
        const frameInfoDisplay = document.getElementById('frameInfo');
        const currentFrameDisplay = document.getElementById('currentFrame');
        const totalFramesDisplay = document.getElementById('totalFrames');
        const directionDisplay = document.getElementById('direction');
        const speedSelect = document.getElementById('speedSelect');
        
        let isPlaying = false;
        let isReverse = false;
        const fps = """ + str(fps) + """;
        
        // Sync all videos
        function syncVideos() {
            const masterVideo = videos[0];
            videos.forEach((video, i) => {
                if (i > 0) {
                    video.currentTime = masterVideo.currentTime;
                }
            });
        }
        
        // Update timeline and displays
        function updateTimeline() {
            const masterVideo = videos[0];
            if (masterVideo.duration) {
                const percentage = (masterVideo.currentTime / masterVideo.duration) * 100;
                timeline.value = percentage;
                
                currentTimeDisplay.textContent = masterVideo.currentTime.toFixed(2) + 's';
                durationDisplay.textContent = masterVideo.duration.toFixed(2) + 's';
                
                const currentFrame = Math.floor(masterVideo.currentTime * fps);
                const totalFrames = Math.floor(masterVideo.duration * fps);
                frameInfoDisplay.textContent = `Frame: ${currentFrame} / ${totalFrames}`;
                currentFrameDisplay.textContent = currentFrame;
                totalFramesDisplay.textContent = totalFrames;
            }
        }
        
        // Play/pause toggle
        function togglePlayPause() {
            if (isPlaying) {
                videos.forEach(v => v.pause());
                playPauseBtn.innerHTML = '▶ Play';
                isPlaying = false;
            } else {
                videos.forEach(v => v.play());
                playPauseBtn.innerHTML = '⏸ Pause';
                isPlaying = true;
            }
        }
        
        // Step forward one frame
        function stepForward() {
            videos.forEach(v => {
                v.pause();
                v.currentTime = Math.min(v.duration, v.currentTime + (1 / fps));
            });
            syncVideos();
            updateTimeline();
            isPlaying = false;
            playPauseBtn.innerHTML = '▶ Play';
        }
        
        // Step backward one frame
        function stepBackward() {
            videos.forEach(v => {
                v.pause();
                v.currentTime = Math.max(0, v.currentTime - (1 / fps));
            });
            syncVideos();
            updateTimeline();
            isPlaying = false;
            playPauseBtn.innerHTML = '▶ Play';
        }
        
        // Reverse playback
        function reversePlayback() {
            isReverse = !isReverse;
            directionDisplay.innerHTML = isReverse ? 'Backward ⬅️' : 'Forward ➡️';
            
            if (isReverse) {
                videos.forEach(v => v.pause());
                playReverseInterval();
            }
        }
        
        let reverseInterval;
        function playReverseInterval() {
            if (reverseInterval) clearInterval(reverseInterval);
            
            if (isReverse && isPlaying) {
                const speed = parseFloat(speedSelect.value);
                reverseInterval = setInterval(() => {
                    videos.forEach(v => {
                        v.currentTime = Math.max(0, v.currentTime - (1 / fps) * speed);
                        if (v.currentTime <= 0) {
                            clearInterval(reverseInterval);
                            isPlaying = false;
                            playPauseBtn.innerHTML = '▶ Play';
                        }
                    });
                    syncVideos();
                    updateTimeline();
                }, 1000 / fps);
            }
        }
        
        // Reset video
        function resetVideo() {
            videos.forEach(v => {
                v.pause();
                v.currentTime = 0;
            });
            syncVideos();
            updateTimeline();
            isPlaying = false;
            playPauseBtn.innerHTML = '▶ Play';
            isReverse = false;
            directionDisplay.innerHTML = 'Forward ➡️';
        }
        
        // Change playback speed
        function changeSpeed() {
            const speed = parseFloat(speedSelect.value);
            videos.forEach(v => v.playbackRate = speed);
        }
        
        // Timeline slider
        timeline.addEventListener('input', function() {
            const masterVideo = videos[0];
            const time = (this.value / 100) * masterVideo.duration;
            videos.forEach(v => v.currentTime = time);
            syncVideos();
            updateTimeline();
        });
        
        // Update timeline as video plays
        videos[0].addEventListener('timeupdate', updateTimeline);
        
        // Initialize when video metadata is loaded
        videos[0].addEventListener('loadedmetadata', function() {
            updateTimeline();
        });
        
        // Sync all videos on time update
        videos[0].addEventListener('timeupdate', syncVideos);
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            switch(e.key) {
                case ' ':
                    e.preventDefault();
                    togglePlayPause();
                    break;
                case 'ArrowLeft':
                    e.preventDefault();
                    stepBackward();
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    stepForward();
                    break;
                case 'r':
                case 'R':
                    e.preventDefault();
                    reversePlayback();
                    break;
                case '0':
                    e.preventDefault();
                    resetVideo();
                    break;
                case '+':
                case '=':
                    e.preventDefault();
                    const currentIndex = speedSelect.selectedIndex;
                    if (currentIndex < speedSelect.options.length - 1) {
                        speedSelect.selectedIndex = currentIndex + 1;
                        changeSpeed();
                    }
                    break;
                case '-':
                case '_':
                    e.preventDefault();
                    const currentIdx = speedSelect.selectedIndex;
                    if (currentIdx > 0) {
                        speedSelect.selectedIndex = currentIdx - 1;
                        changeSpeed();
                    }
                    break;
                case 'f':
                case 'F':
                    e.preventDefault();
                    if (videos[0].requestFullscreen) {
                        videos[0].requestFullscreen();
                    }
                    break;
            }
        });
        
        // Handle reverse playback
        videos.forEach(v => {
            v.addEventListener('play', function() {
                if (isReverse) {
                    playReverseInterval();
                }
            });
            v.addEventListener('pause', function() {
                if (reverseInterval) clearInterval(reverseInterval);
            });
        });
    </script>
</body>
</html>"""
    
    with open(output_html, 'w') as f:
        f.write(html_content)
    
    print(f"✓ Created interactive video viewer: {output_html}")
    print(f"  Videos: {len(video_paths)}")
    print(f"  Layout: {grid_cols} columns")
    print(f"\n🌐 Open the HTML file in your browser to view!")


def main():
    parser = argparse.ArgumentParser(description='Create interactive HTML video viewer')
    parser.add_argument('videos', nargs='+', help='Path(s) to video file(s)')
    parser.add_argument('-o', '--output', default='video_viewer.html', help='Output HTML file')
    parser.add_argument('-t', '--titles', nargs='+', help='Titles for each video')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    
    args = parser.parse_args()
    
    # Verify all videos exist
    for video_path in args.videos:
        if not os.path.exists(video_path):
            print(f"Error: Video not found: {video_path}")
            return 1
    
    create_video_viewer_html(args.videos, args.output, args.titles, args.fps)
    return 0


if __name__ == '__main__':
    exit(main())

