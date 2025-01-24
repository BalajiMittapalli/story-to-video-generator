from typing import List, Optional
from dotenv import load_dotenv
import os
import requests
import time

load_dotenv()

REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")
REPLICATE_API_URL = "https://api.replicate.com/v1/predictions"
REPLICATE_MODEL = "ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4"
IMAGE_HEIGHT = 768
IMAGE_WIDTH = 1024
GENERATIONS_DIR = "generations"


class Scene:
    def __init__(self, index: int, imagePrompt: str, audioText: str, imageFile: Optional[str] = None, audioFile: Optional[str] = None):
        self.index = index
        self.imagePrompt = imagePrompt
        self.audioText = audioText
        self.imageFile = imageFile
        self.audioFile = audioFile


class Project:
    def __init__(self, name: str, story: str, scenes: List[Scene]):
        self.name = name
        self.story = story
        self.scenes = scenes


def create_scenes():
    pass


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

            if response.status_code == 200:
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
        'Prefer': 'wait'
    }

    project_dir = os.path.join(GENERATIONS_DIR, project.name.replace(" ", "-"))
    os.makedirs(project_dir, exist_ok=True)

    index = 1

    for scene in project.scenes:
        body = {
            "version": REPLICATE_MODEL,
            "input": {
                "width": IMAGE_WIDTH,
                "height": IMAGE_HEIGHT,
                "prompt": scene.imagePrompt,
                "scheduler": "K_EULER",
                "num_outputs": 1,
                "guidance_scale": 7.5,
                "num_inference_steps": 50
            }
        }

        try:
            request = requests.post(
                url=REPLICATE_API_URL,
                headers=headers,
                json=body
            )

            response = request.json()
            print("Response:", response)

            prediction_id = response.get("id")
            print("Prediction ID:", prediction_id)
            if prediction_id:
                output_url = poll_for_completion(prediction_id)
                print("Output URL:", output_url)
                if output_url:
                    image_path = os.path.join(project_dir, f"{index}.png")
                    image_response = requests.get(output_url)

                    if image_response.status_code == 200:
                        with open(image_path, "wb") as image_file:
                            image_file.write(image_response.content)
                        print(f"Image saved to: {image_path}")

                        scene.imageFile = image_path
                        index += 1
                    else:
                        print(f"Failed to download image: {
                              image_response.status_code}")
                else:
                    print("Prediction did not complete successfully.")
            else:
                print("No prediction ID found in the response.")

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")


def create_audio():
    pass


def image_to_video_with_captions_overlay_and_audio():
    pass


def merge_video_scenes():
    pass


def main():
    project = Project(
        name="My Story",
        story="A story about a boy and a car.",
        scenes=[
            # Scene(index=0, imagePrompt="a boy taking water bottle from fridge at night",
            #       audioText="", imageFile=None, audioFile=None),
            Scene(index=1, imagePrompt="a futuristic city at sunset",
                  audioText="", imageFile=None, audioFile=None),
        ]
    )

    create_images(project)


if __name__ == "__main__":
    os.makedirs(GENERATIONS_DIR, exist_ok=True)
    main()
