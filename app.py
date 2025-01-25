from typing import List, Optional
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

from moviepy.editor import ImageClip, TextClip, CompositeVideoClip, AudioFileClip
from moviepy.config import change_settings

from dotenv import load_dotenv
from gtts import gTTS
import os
import requests
import json
import time

load_dotenv()


REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")
REPLICATE_API_URL = "https://api.replicate.com/v1/predictions"
REPLICATE_MODEL = "ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4"
IMAGE_HEIGHT = 768
IMAGE_WIDTH = 1024
GENERATIONS_DIR = "generations"

change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})

llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.5,
)

prompt_template = PromptTemplate(
    input_variables=["name", "story"],
    template="""
                
                You are a professional story writer for short form video content where the video will be created with a maximum of 10 scenes. You write the scenes for the story given by the user. The video that will be generated will comprise of images with animations and audio telling the story.

                The scene will be an array which will consist of image_prompt and audio_text of all the scenes like a json file of each scene.
                You need to write the audio_text of the scene in such a way that it will be a continuation of the previous scene audio_text so that the story will be continuous. The images need not be continous but should atleast be a little related to the scene.

                The scene will consist of a image with small animations like zoom in, zoom out, pan, etc. The audio will be a voice over of the text of the scene.
                The image that will be displayed will be generated by ai with the image_prompt from the scene. The audio will be generated from the audio_text from the scene.

                Each scene should not be too long and each scene will be of the length of the audio_text. That is you need to generate the audio_text of each scene with 10-15 words and not more, so structure the scenes accordingly.

                You as the scene writer for the story needs to write a description for the image to be generated by ai and also you need to wite short audio overlays for the scene being delivered at the moment.
                So you need to think about the how many scenes should be there and how to structure them properly and how you need to write the image_prompt for the scene and how you need to write the audio_text for the scene.

                The output should be a json object with the scenes array which will consist of image_prompt and audio_text of all the scenes.
                
                Ensure the response is valid JSON and can be parsed by Python's json.loads() function.
                Do not include any additional text outside the JSON object.
                

                name of the story given by the user: {name}
                story given by the user: {story}
                Answer:""")


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


def create_scenes(project: Project, prompt_template):
    scenes_json_path = os.path.join(project.project_dir, "scenes.json")

    # Check if scenes.json already exists
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
                scenes.append(scene)

            project.scenes = scenes
            print(f"Loaded existing scenes from {scenes_json_path}")
            return  # Exit early since we loaded existing scenes
        except Exception as e:
            print(f"Error loading scenes.json: {e}. Generating new scenes.")

    # Existing generation code if no scenes.json found
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

        # Update the project's scenes
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
        
        # Check if image already exists
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


def create_audio(project: Project):
    audio_dir = os.path.join(project.project_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    for scene in project.scenes:
        if not scene.audio_text:
            continue

        audio_path = os.path.join(audio_dir, f"{scene.index}.mp3")
        
        # Check for existing audio file
        if os.path.exists(audio_path):
            scene.audio_file = audio_path
            print(f"Audio for scene {scene.index} already exists. Skipping generation.")
            continue

        try:
            tts = gTTS(text=scene.audio_text, lang='en', slow=False)
            tts.save(audio_path)
            scene.audio_file = audio_path
            print(f"Generated audio for scene {scene.index}")
        except Exception as e:
            print(f"Error generating audio for scene {scene.index}: {str(e)}")


def create_scene_videos(project: Project):
    videos_dir = os.path.join(project.project_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    for scene in project.scenes:
        if not scene.image_file or not scene.audio_file:
            print(f"Skipping video creation for scene {
                  scene.index} - missing assets")
            continue

        try:
            # Load assets
            audio_clip = AudioFileClip(scene.audio_file)
            image_clip = ImageClip(scene.image_file).set_duration(
                audio_clip.duration)

            # Create text overlay
            text_clip = (
                TextClip(
                    scene.audio_text,
                    fontsize=28,
                    color="white",
                    font="Arial-Bold",
                    stroke_color="black",
                    stroke_width=1,
                )
                .set_position(("center", "center"))
                .set_duration(audio_clip.duration)
            )

            # Create composite video
            final_clip = CompositeVideoClip([image_clip, text_clip])
            final_clip = final_clip.set_audio(audio_clip)

            # Write output
            video_path = os.path.join(videos_dir, f"{scene.index}.mp4")
            final_clip.write_videofile(
                video_path,
                fps=24,
                codec="libx264",
                audio_codec="aac",
                threads=4,
                logger=None,
            )

            print(f"Created video for scene {scene.index} at {video_path}")

        except Exception as e:
            print(f"Error creating video for scene {scene.index}: {str(e)}")
        finally:
            # Cleanup to prevent memory issues
            if 'final_clip' in locals():
                final_clip.close()
            if 'audio_clip' in locals():
                audio_clip.close()
            if 'image_clip' in locals():
                image_clip.close()


def merge_video_scenes():
    pass


def main():

    name = input("Enter the name of your story: ")
    story = input("Enter the story: ")

    project = Project(name=name, story=story, scenes=[])

    create_scenes(project, prompt_template)
    # for scene in project.scenes:
    #     print(scene.index, scene.image_prompt, scene.audio_text)
    # create_images(project)
    # create_audio(project)
    # create_scene_videos(project)


if __name__ == "__main__":
    os.makedirs(GENERATIONS_DIR, exist_ok=True)
    main()


# pip install moviepy imagemagick