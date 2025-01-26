# Create an AI tool that transforms regional folk stories into animated short
# films with visuals and regional voiceovers, preserving local dialects, local art styles and cultural nuances.

from typing import List, Optional
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

from moviepy import ImageClip, CompositeVideoClip, VideoFileClip, concatenate_videoclips
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.fx import Resize

from dotenv import load_dotenv
from gtts import gTTS
import os
import requests
import json
import time

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-1.5-flash"
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")
REPLICATE_API_URL = "https://api.replicate.com/v1/predictions"  # api endpoint
# model id - stable diffusion
REPLICATE_MODEL = "ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4"
IMAGE_HEIGHT = 768
IMAGE_WIDTH = 1024
GENERATIONS_DIR = "generations"


llm = GoogleGenerativeAI(
    model=GEMINI_MODEL,
    api_key=GEMINI_API_KEY,
    temperature=0.5,
)


prompt_template = PromptTemplate(
    input_variables=["name", "story", "language"],
    template="""
    You are a cultural preservation specialist creating animated short films from Indian regional folk stories. 
    Generate video scenes that maintain:
    1. Authentic local dialects in voiceovers
    2. Traditional Indian art styles (Warli, Madhubani, Pattachitra, etc.)
    3. Cultural nuances and historical context

    Structure requirements:
    - Maximum 7 scenes
    - Each scene contains:
      * image_prompt: Visual description including:
        - Specific Indian art style reference
        - Traditional clothing and jewelry
        - Regional architecture/landscapes
        - Cultural symbols/motifs
        - Character actions/emotions
      * audio_text: Dialogue/narration in regional language (10-12 words) with local idioms
    - Image and audio text should correspond to create a cohesive narrative in generated scenes.

    Image Prompt Guidelines:
    - Specify art style: "Warli tribal art with white-on-red motifs", "Madhubani-style figures with intricate floral borders"
    - Include cultural elements: "Nataraja sculpture in background", "Women in Kanjeevaram sarees performing Kolattam"
    - Add movement cues: "Villagers dancing in circle", "Farmer ploughing field at sunrise"
    - Use regional color palettes: "Tanjore painting gold leaf accents", "Rajasthani fresco colors"

    Audio Guidelines:
    - Use regional linguistic features: Tamil honorifics, Bengali alliteration
    - Maintain oral storytelling rhythm
    - Include traditional song/stanza where appropriate
    - Generate the audio text in only 1 language: Telugu (te) or Hindi (hi) as string
    - Give only in the specified language don't give in english.
    - Give only once for each scene

    Output: JSON object with scenes array containing image_prompt and audio_text.
    Ensure valid JSON format parseable by Python's json.loads().
    No text outside JSON.

    Story Name: {name}
    Folk Story Content: {story}
    Language: {language}
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
    def __init__(self, name: str, story: str, language: str, scenes: List[Scene]):
        self.name = name
        self.story = story
        self.language = language
        self.scenes = scenes

    @property
    def project_dir(self):
        return os.path.join(GENERATIONS_DIR, self.name.replace(" ", "-"))


def get_project(name: str, story: str, language: str) -> Project:
    project_dir = os.path.join(GENERATIONS_DIR, name.replace(" ", "-"))
    scenes_json_path = os.path.join(project_dir, "scenes.json")

    if not os.path.exists(scenes_json_path):
        os.makedirs(project_dir, exist_ok=True)
        print(f"Created new project directory: {project_dir}")
        return Project(name=name, story=story, language=language, scenes=[])

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

    return Project(name=name, story=story, language=language, scenes=scenes)


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
            "story": project.story,
            "language": project.language
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
                    print(f"Prediction status: {status}. Polling again in 5 seconds...")
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
            print(f"Image for scene {scene.index} already exists. Skipping generation.")
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
    combined_text = " ".join(
        scene.audio_text for scene in project.scenes if scene.audio_text)
    combined_audio_path = os.path.join(audio_dir, "combined.mp3")
    try:
        tts = gTTS(text=combined_text, lang=project.language, slow=False)
        tts.save(combined_audio_path)
        print("Generated combined audio for all scenes.")
    except Exception as e:
        print(f"Error generating combined audio: {str(e)}")
    for scene in project.scenes:
        if not scene.audio_text:
            continue

        audio_path = os.path.join(audio_dir, f"{scene.index}.mp3")

        if os.path.exists(audio_path):
            scene.audio_file = audio_path
            print(f"Audio for scene {scene.index} already exists. Skipping generation.")
            continue

        try:
            tts = gTTS(text=scene.audio_text, lang=project.language, slow=False)
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
    videos_dir = os.path.join(project.project_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    for scene in project.scenes:
        if not scene.image_file or not scene.audio_file:
            print(f"Skipping video creation for scene {scene.index} - missing assets")
            continue

        try:
            audio_clip = AudioFileClip(scene.audio_file)
            audio_duration = audio_clip.duration
            audio_clip.close()

            def zoom_factor(t):
                return 1.0 + (t / audio_duration) * 0.2 

            img_clip = ImageClip(
                scene.image_file).with_duration(audio_duration)

            final_clip = (
                img_clip
                .with_effects([
                    Resize(new_size=lambda t: (
                        int(img_clip.w * zoom_factor(t)),
                        int(img_clip.h * zoom_factor(t))
                    ))
                ])
            )

            video_clip = CompositeVideoClip(
                [final_clip],
                size=(IMAGE_WIDTH, IMAGE_HEIGHT)
            ).with_duration(audio_duration)

            video_path = os.path.join(videos_dir, f"{scene.index}.mp4")
            video_clip.write_videofile(
                video_path,
                fps=24,
                codec="libx264",
                audio_codec="aac",
                threads=4,
                preset='medium',
                ffmpeg_params=[
                    '-crf', '18',
                    '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart'
                ]
            )

            print(f"Created dynamic video for scene {scene.index}")

        except Exception as e:
            print(f"Error creating video for scene {scene.index}: {str(e)}")
        finally:
            if 'img_clip' in locals():
                img_clip.close()
            if 'video_clip' in locals():
                video_clip.close()


def create_final_scenes(project: Project):
    """Create individual scene videos with merged audio"""
    try:
        # Setup paths
        scene_videos_dir = os.path.join(project.project_dir, "videos")
        audio_dir = os.path.join(project.project_dir, "audio")
        final_scenes_dir = os.path.join(project.project_dir, "final_scenes")
        os.makedirs(final_scenes_dir, exist_ok=True)

        # Process each scene
        for scene in project.scenes:
            if not scene.audio_file or not scene.image_file:
                print(f"Skipping scene {scene.index} - missing assets")
                continue

            # Path setup
            scene_video_path = os.path.join(
                scene_videos_dir, f"{scene.index}.mp4")
            final_scene_path = os.path.join(
                final_scenes_dir, f"{scene.index}.mp4")

            if os.path.exists(final_scene_path):
                print(f"Final scene {scene.index} already exists. Skipping.")
                continue

            # Load assets
            video_clip = VideoFileClip(scene_video_path)
            audio_clip = AudioFileClip(scene.audio_file)

            try:
                # Synchronize durations
                if abs(video_clip.duration - audio_clip.duration) > 0.1:
                    print(f"Adjusting scene {scene.index} duration (V: {video_clip.duration:.2f}s, A: {audio_clip.duration:.2f}s)")
                    video_clip = video_clip.subclip(0, audio_clip.duration)

                # Combine audio and video
                final_clip = video_clip.with_audio(audio_clip)

                # Write final scene video
                final_clip.write_videofile(
                    final_scene_path,
                    fps=24,
                    codec="libx264",
                    audio_codec="aac",
                    audio_bitrate="192k",
                    threads=2,
                    preset="medium",
                    ffmpeg_params=[
                        '-crf', '18',
                        '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart'
                    ]
                )
                print(f"Created final scene {scene.index} with audio")

            finally:
                video_clip.close()
                audio_clip.close()

    except Exception as e:
        print(f"Failed to create final scenes: {str(e)}")
        raise


def merge_final_scenes(project: Project):
    """Merge all final scene videos into complete film"""
    try:
        final_scenes_dir = os.path.join(project.project_dir, "final_scenes")
        output_path = os.path.join(project.project_dir, "final.mp4")

        # Get sorted final scenes
        scene_files = sorted(
            [f for f in os.listdir(final_scenes_dir) if f.endswith(".mp4")],
            key=lambda x: int(x.split(".")[0])
        )

        if not scene_files:
            raise ValueError("No final scene videos found to merge")

        # Load and concatenate scenes
        clips = [VideoFileClip(os.path.join(final_scenes_dir, f))
                 for f in scene_files]
        final_film = concatenate_videoclips(clips, method="compose")

        # Write final output
        final_film.write_videofile(
            output_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            audio_bitrate="192k",
            threads=4,
            preset="medium",
            ffmpeg_params=[
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart'
            ]
        )
        print(f"Successfully created final film: {output_path}")

    except Exception as e:
        print(f"Failed to merge final scenes: {str(e)}")
        raise
    finally:
        for clip in clips:
            clip.close()
        if 'final_film' in locals():
            final_film.close()


def main():
    name = input("Enter the name of your story: ")
    story = input("Enter the story: ")
    language = input("Enter the language (te for Telugu, hi for Hindi): ")

    if language not in ["te", "hi"]:
        print("Invalid language selection. Please choose 'te' for Telugu or 'hi' for Hindi.")
        return

    project = get_project(name, story, language)

    create_scenes(project, prompt_template)
    # create_images(project)
    create_audio(project)
    create_scene_videos(project)
    create_final_scenes(project)
    merge_final_scenes(project)


if __name__ == "__main__":
    os.makedirs(GENERATIONS_DIR, exist_ok=True)
    main()