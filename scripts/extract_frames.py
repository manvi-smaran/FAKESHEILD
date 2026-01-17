import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    fps: int = 1,
    max_frames: int = 10,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_pattern = str(output_dir / f"{video_path.stem}_%04d.jpg")
    
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-frames:v", str(max_frames),
        "-q:v", "2",
        "-y",
        output_pattern,
    ]
    
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    
    return len(list(output_dir.glob(f"{video_path.stem}_*.jpg")))


def process_celebdf(
    input_dir: Path,
    output_dir: Path,
    fps: int = 1,
    max_frames: int = 10,
    num_workers: int = 4,
):
    folders = {
        "Celeb-real": ("Celeb-real", 0),
        "Celeb-synthesis": ("Celeb-synthesis", 1),
        "YouTube-real": ("YouTube-real", 0),
    }
    
    for folder_name, (out_name, label) in folders.items():
        folder_path = input_dir / folder_name
        if not folder_path.exists():
            print(f"Skipping {folder_name} (not found)")
            continue
        
        videos = list(folder_path.glob("*.mp4"))
        print(f"\nProcessing {folder_name}: {len(videos)} videos")
        
        out_folder = output_dir / out_name
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for video in videos:
                video_out_dir = out_folder / video.stem
                future = executor.submit(
                    extract_frames_from_video,
                    video,
                    video_out_dir,
                    fps,
                    max_frames,
                )
                futures[future] = video.name
            
            total_frames = 0
            for future in tqdm(as_completed(futures), total=len(futures)):
                video_name = futures[future]
                try:
                    n_frames = future.result()
                    total_frames += n_frames
                except Exception as e:
                    print(f"Error processing {video_name}: {e}")
        
        print(f"  Extracted {total_frames} frames → {out_folder}")


def process_faceforensics(
    input_dir: Path,
    output_dir: Path,
    fps: int = 1,
    max_frames: int = 10,
    num_workers: int = 4,
):
    folders = [
        "original",
        "Deepfakes", 
        "Face2Face",
        "FaceSwap",
        "FaceShifter",
        "NeuralTextures",
        "DeepFakeDetection",
    ]
    
    for folder_name in folders:
        folder_path = input_dir / folder_name
        if not folder_path.exists():
            print(f"Skipping {folder_name} (not found)")
            continue
        
        videos = list(folder_path.glob("**/*.mp4"))
        print(f"\nProcessing {folder_name}: {len(videos)} videos")
        
        out_folder = output_dir / folder_name
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for video in videos:
                video_out_dir = out_folder / video.stem
                future = executor.submit(
                    extract_frames_from_video,
                    video,
                    video_out_dir,
                    fps,
                    max_frames,
                )
                futures[future] = video.name
            
            total_frames = 0
            for future in tqdm(as_completed(futures), total=len(futures)):
                video_name = futures[future]
                try:
                    n_frames = future.result()
                    total_frames += n_frames
                except Exception as e:
                    print(f"Error processing {video_name}: {e}")
        
        print(f"  Extracted {total_frames} frames → {out_folder}")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from deepfake datasets")
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["celebdf", "faceforensics", "ff"],
        help="Dataset type to process",
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to dataset root directory",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for extracted frames (default: ./data/<dataset>)",
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Frames per second to extract (default: 1)",
    )
    
    parser.add_argument(
        "--max_frames",
        type=int,
        default=10,
        help="Maximum frames per video (default: 10)",
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    
    args = parser.parse_args()
    
    if args.output is None:
        if args.dataset == "celebdf":
            args.output = "./data/celeb_df"
        else:
            args.output = "./data/faceforensics"
    
    print("=" * 50)
    print("Deepfake Dataset Frame Extraction")
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Input:   {args.input}")
    print(f"Output:  {args.output}")
    print(f"FPS:     {args.fps}")
    print(f"Max frames/video: {args.max_frames}")
    print("=" * 50)
    
    if args.dataset == "celebdf":
        process_celebdf(
            input_dir=Path(args.input),
            output_dir=Path(args.output),
            fps=args.fps,
            max_frames=args.max_frames,
            num_workers=args.workers,
        )
    else:
        process_faceforensics(
            input_dir=Path(args.input),
            output_dir=Path(args.output),
            fps=args.fps,
            max_frames=args.max_frames,
            num_workers=args.workers,
        )
    
    print("\n✓ Frame extraction complete!")


if __name__ == "__main__":
    main()
