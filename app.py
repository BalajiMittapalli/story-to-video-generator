# Create an AI tool that transforms regional folk stories into animated short
# films with visuals and regional voiceovers, preserving local dialects, local art styles and cultural nuances.

from typing import List, Optional
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

from moviepy import ImageClip, CompositeVideoClip, VideoFileClip, concatenate_videoclips
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import concatenate_audioclips

from dotenv import load_dotenv
from gtts import gTTS
import os
import requests
import json
import time

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")
REPLICATE_API_URL = "https://api.replicate.com/v1/predictions"  # replicate api endpoint
# this is not api key, this is model id, dont waste ur time testing it
REPLICATE_MODEL = "ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4"
GEMINI_MODEL = "gemini-1.5-flash"
IMAGE_HEIGHT = 768
IMAGE_WIDTH = 1024
GENERATIONS_DIR = "generations"


llm = GoogleGenerativeAI(
    model=GEMINI_MODEL,
    api_key=GEMINI_API_KEY,
    temperature=0.5,
)


prompt_template = PromptTemplate(
    input_variables=["name", "story"],
    template="""
    You are a cultural preservation specialist creating animated short films from regional folk stories. 
    Generate video scenes that maintain:
    1. Authentic local dialects in voiceovers
    2. Traditional art styles in visuals
    3. Cultural nuances and historical context

    Structure requirements:
    - Maximum 3 scenes
    - Each scene contains:
      * image_prompt: Visual description incorporating regional art elements (traditional clothing, architecture, natural landscapes)
      * audio_text: Dialogue/narration in regional dialect (10-15 words) preserving local idioms and speech patterns

    Guidelines:
    1. Image Prompts Should:
    - Reference specific regional art characteristics (e.g., "Warli-style tribal patterns", "Madhubani folk art elements")
    - Include culturally significant colors, textures, and motifs
    - Depict traditional settings and historical accuracy

    2. Audio Text Must:
    - Use authentic regional vocabulary and sentence structures
    - Preserve oral storytelling traditions
    - Maintain natural flow between scenes

    Output: JSON object with scenes array containing image_prompt and audio_text.
    Ensure valid JSON format parseable by Python's json.loads().
    No text outside JSON.

    Story Name: {name}
    Folk Story Content: {story}
    Answer:"""
)


class Scene:
    def __init__(self):
        self.index: int = 0
        self.image_prompt: str = ""
        self.audio_text: str = ""
        self.image_file: Optional[str] = None
        self.audio_file: Optional[str] = None


class Project:
    def __init__(self, name: str, story: str, scenes: List[Scene]):
        self.name = name
        self.story = story
        self.scenes = scenes

    @property
    def project_dir(self):
        return os.path.join(GENERATIONS_DIR, self.name.replace(" ", "-"))


def get_project(name: str, story: str) -> Project:
    project_dir = os.path.join(GENERATIONS_DIR, name.replace(" ", "-"))
    scenes_json_path = os.path.join(project_dir, "scenes.json")

    if not os.path.exists(scenes_json_path):
        os.makedirs(project_dir, exist_ok=True)
        print(f"Created new project directory: {project_dir}")
        return Project(name=name, story=story, scenes=[])

    with open(scenes_json_path, "r") as f:
        scenes_data = json.load(f)

    scenes = []
    for scene_data in scenes_data:
        scene = Scene()
        scene.index = scene_data["index"]
        scene.image_prompt = scene_data["image_prompt"]
        scene.audio_text = scene_data["audio_text"]
        scene.image_file = scene_data.get("image_file")
        scene.audio_file = scene_data.get("audio_file")
        scenes.append(scene)

    return Project(name=name, story=story, scenes=scenes)


def create_scenes(project: Project, prompt_template):
    scenes_json_path = os.path.join(project.project_dir, "scenes.json")

    if os.path.exists(scenes_json_path):
        try:
            with open(scenes_json_path, "r") as f:
                scenes_data = json.load(f)

            scenes = []
            for scene_data in scenes_data:
                scene = Scene()
                scene.index = scene_data["index"]
                scene.image_prompt = scene_data["image_prompt"]
                scene.audio_text = scene_data["audio_text"]
                scene.image_file = scene_data.get("image_file")
                scene.audio_file = scene_data.get("audio_file")
                scenes.append(scene)

            project.scenes = scenes
            print(f"Loaded existing scenes from {scenes_json_path}")
            return
        except Exception as e:
            print(f"Error loading scenes.json: {e}. Generating new scenes.")

    chain = LLMChain(llm=llm, prompt=prompt_template)

    try:
        response = chain.invoke({
            "name": project.name,
            "story": project.story
        })

        raw_response = response['text'].strip()

        if raw_response.startswith("```json"):
            raw_response = raw_response[7:-3].strip()

        response_json = json.loads(raw_response)
        scenes_data = response_json.get("scenes", [])

        scenes = []
        for idx, scene_data in enumerate(scenes_data, start=1):
            scene = Scene()
            scene.index = idx
            scene.image_prompt = scene_data.get("image_prompt", "")
            scene.audio_text = scene_data.get("audio_text", "")
            scenes.append(scene)

        project.scenes = scenes

        os.makedirs(project.project_dir, exist_ok=True)
        scenes_json = [
            {
                "index": scene.index,
                "image_prompt": scene.image_prompt,
                "audio_text": scene.audio_text
            }
            for scene in project.scenes
        ]
        with open(scenes_json_path, "w") as f:
            json.dump(scenes_json, f, indent=4)

        print(f"Created project with {len(scenes)} scenes:")
        for scene in project.scenes:
            print(f"Scene {scene.index}:")
            print(f"Image prompt: {scene.image_prompt}")
            print(f"Audio text: {scene.audio_text}\n")

    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
    except Exception as e:
        print(f"Error processing response: {e}")


def poll_for_completion(prediction_id: str) -> Optional[str]:
    headers = {
        'Authorization': 'Bearer ' + REPLICATE_API_KEY,
        'Content-Type': 'application/json',
    }

    while True:
        try:
            response = requests.get(
                f"{REPLICATE_API_URL}/{prediction_id}",
                headers=headers
            )

            if response.status_code == 200 or response.status_code == 201:
                prediction = response.json()
                status = prediction.get("status")

                if status == "succeeded":
                    return prediction.get("output", [None])[0]
                elif status in ["failed", "canceled"]:
                    print(f"Prediction failed or was canceled: {prediction}")
                    return None
                else:
                    print(f"Prediction status: {
                          status}. Polling again in 5 seconds...")
                    time.sleep(5)
            else:
                print(f"Error polling prediction: {response.status_code}")
                print(response.text)
                return None

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None


def create_images(project: Project):
    headers = {
        'Authorization': 'Bearer ' + REPLICATE_API_KEY,
        'Content-Type': 'application/json',
    }

    images_dir = os.path.join(project.project_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    for scene in project.scenes:
        if not scene.image_prompt:
            continue

        image_path = os.path.join(images_dir, f"{scene.index}.png")

        if os.path.exists(image_path):
            scene.image_file = image_path
            print(f"Image for scene {
                  scene.index} already exists. Skipping generation.")
            continue

        body = {
            "version": REPLICATE_MODEL,
            "input": {
                "width": IMAGE_WIDTH,
                "height": IMAGE_HEIGHT,
                "prompt": scene.image_prompt,
                "scheduler": "K_EULER",
                "num_outputs": 1,
                "guidance_scale": 7.5,
                "num_inference_steps": 50
            }
        }

        try:
            response = requests.post(
                url=REPLICATE_API_URL,
                headers=headers,
                json=body
            )
            response.raise_for_status()
            prediction = response.json()
            prediction_id = prediction.get("id")

            if output_url := poll_for_completion(prediction_id):
                img_response = requests.get(output_url)
                img_response.raise_for_status()

                with open(image_path, "wb") as f:
                    f.write(img_response.content)

                scene.image_file = image_path
                print(f"Generated image for scene {scene.index}")
            else:
                print(f"Failed to generate image for scene {scene.index}")

        except Exception as e:
            print(f"Error generating image for scene {scene.index}: {str(e)}")

    # Update scenes.json with image_file paths
    scenes_json = []
    for scene in project.scenes:
        scene_data = {
            "index": scene.index,
            "image_prompt": scene.image_prompt,
            "audio_text": scene.audio_text,
            "image_file": scene.image_file,
            "audio_file": scene.audio_file
        }
        scenes_json.append(scene_data)

    scenes_json_path = os.path.join(project.project_dir, "scenes.json")
    with open(scenes_json_path, "w") as f:
        json.dump(scenes_json, f, indent=4)
    print("Updated scenes.json with image file paths")


def create_audio(project: Project):
    audio_dir = os.path.join(project.project_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    for scene in project.scenes:
        if not scene.audio_text:
            continue

        audio_path = os.path.join(audio_dir, f"{scene.index}.mp3")

        if os.path.exists(audio_path):
            scene.audio_file = audio_path
            print(f"Audio for scene {
                  scene.index} already exists. Skipping generation.")
            continue

        try:
            tts = gTTS(text=scene.audio_text, lang='hi', slow=False)
            tts.save(audio_path)
            scene.audio_file = audio_path
            print(f"Generated audio for scene {scene.index}")
        except Exception as e:
            print(f"Error generating audio for scene {scene.index}: {str(e)}")

    # Update scenes.json with audio_file paths
    scenes_json = []
    for scene in project.scenes:
        scene_data = {
            "index": scene.index,
            "image_prompt": scene.image_prompt,
            "audio_text": scene.audio_text,
            "image_file": scene.image_file,
            "audio_file": scene.audio_file
        }
        scenes_json.append(scene_data)

    scenes_json_path = os.path.join(project.project_dir, "scenes.json")
    with open(scenes_json_path, "w") as f:
        json.dump(scenes_json, f, indent=4)
    print("Updated scenes.json with audio file paths")


def create_scene_videos(project: Project):
    """Create silent videos with duration matching audio files"""
    videos_dir = os.path.join(project.project_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    for scene in project.scenes:
        if not scene.image_file or not scene.audio_file:
            print(f"Skipping video creation for scene {
                  scene.index} - missing assets")
            continue

        image_clip = None
        audio_clip = None
        video_clip = None

        try:
            # Load audio to get duration
            audio_clip = AudioFileClip(scene.audio_file)

            # Create image clip with audio duration
            image_clip = ImageClip(scene.image_file).with_duration(
                audio_clip.duration)

            # Create silent video clip
            video_clip = CompositeVideoClip(
                [image_clip.with_position('center')],
                size=image_clip.size
            ).with_duration(audio_clip.duration)

            # Write video without audio
            video_path = os.path.join(videos_dir, f"{scene.index}.mp4")
            video_clip.write_videofile(
                video_path,
                fps=24,
                codec="libx264",
                threads=4,
                preset='medium',
                ffmpeg_params=[
                    '-crf', '18',
                    '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart'
                ]
            )

            print(f"Created silent video for scene {scene.index}")

        except Exception as e:
            print(f"Error creating video for scene {scene.index}: {str(e)}")

        finally:
            if video_clip:
                video_clip.close()
            if image_clip:
                image_clip.close()
            if audio_clip:
                audio_clip.close()


def merge_audio_files(project: Project) -> str:
    audio_dir = os.path.join(project.project_dir, "audio")
    output_path = os.path.join(project.project_dir, "merged_audio.wav")

    # Get sorted audio files by scene index
    audio_files = sorted([
        (scene.index, scene.audio_file) 
        for scene in project.scenes 
        if scene.audio_file and os.path.exists(scene.audio_file)
    ], key=lambda x: x[0])

    if not audio_files:
        raise ValueError("No valid audio files found to merge")

    audio_clips = []
    total_duration = 0.0
    try:
        # Load and verify all audio clips
        for index, audio_file in audio_files:
            clip = AudioFileClip(audio_file)
            if clip.fps != 44100:
                clip = clip.with_fps(44100)
            audio_clips.append(clip)
            total_duration += clip.duration
            print(f"Scene {index}: {clip.duration:.2f}s")

        # Concatenate with 0.5s crossfade between clips
        final_audio = concatenate_audioclips(audio_clips)
        final_duration = final_audio.duration
        
        print(f"Original duration: {total_duration:.2f}s")
        print(f"Merged duration: {final_duration:.2f}s (with crossfades)")

        # Write final audio
        final_audio.write_audiofile(
            output_path,
            fps=44100,
            codec='pcm_s16le',
            verbose=False
        )
        return output_path

    finally:
        for clip in audio_clips:
            clip.close()
        if 'final_audio' in locals():
            final_audio.close()

def merge_video_scenes(project: Project):
    videos_dir = os.path.join(project.project_dir, "videos")
    output_path = os.path.join(project.project_dir, "final.mp4")

    # Load video clips
    scene_files = sorted(
        [f for f in os.listdir(videos_dir) if f.endswith(".mp4")],
        key=lambda x: int(x.split(".")[0])
    )
    
    clips = [VideoFileClip(os.path.join(videos_dir, f)) for f in scene_files]
    video_duration = sum(clip.duration for clip in clips)
    
    try:
        # Merge videos
        final_video = concatenate_videoclips(clips, method="compose")
        print(f"Video duration: {final_video.duration:.2f}s")

        # Merge audio
        merged_audio_path = merge_audio_files(project)
        final_audio = AudioFileClip(merged_audio_path)
        print(f"Audio duration: {final_audio.duration:.2f}s")

        # Synchronize durations
        duration_diff = final_video.duration - final_audio.duration
        if abs(duration_diff) > 0.1:
            print(f"Adjusting video duration by {duration_diff:.2f}s")
            final_video = final_video.with_duration(final_audio.duration)

        # Combine audio and video
        final_video = final_video.with_audio(final_audio)
        
        # Write final file
        final_video.write_videofile(
            output_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            audio_bitrate="192k",
            threads=4,
            logger=None
        )
        print(f"Final video created: {output_path}")

    finally:
        for clip in clips:
            clip.close()
        if 'final_video' in locals():
            final_video.close()
        if 'final_audio' in locals():
            final_audio.close()
        if os.path.exists(merged_audio_path):
            os.remove(merged_audio_path)

def main():
    name = input("Enter the name of your story: ")
    story = input("Enter the story: ")

    project = get_project(name, story)

    # create_scenes(project, prompt_template)
    # create_images(project)
    # create_audio(project)
    create_scene_videos(project)
    merge_video_scenes(project)


if __name__ == "__main__":
    os.makedirs(GENERATIONS_DIR, exist_ok=True)
    main()
