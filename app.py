from typing import List


class Scene:
    index: int
    imagePrompt: str
    audioText: str
    imageFile: str | None
    audioFile: str | None


class Project:
    def __init__(self, name: str, story: str, scenes: List[Scene]):
        self.name = name
        self.story = story
        self.scenes = scenes


def create_scenes(prompt_template):
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.invoke({
                "name": "fear at night",
                "story": "A boy sleeping on his bed at night wanted to drink water, so he went to fridge and while taking water from the fridge something fell on top of him, so he ran to his bed but felt that something maybe sleeping in his sheets. so he went under the bed to sleep but then there is a hgost there, and then he suddenly woke from his dream and realised that it was just a dream."
            })
    
    raw_response = response['text'].strip()
    

    if raw_response.startswith("```json"):
        raw_response = raw_response[7:-3].strip()
    
    try:
        response_json = json.loads(raw_response)
        scenes_data = response_json.get("scenes", [])


        scenes = []
        for idx, scene_data in enumerate(scenes_data, start=1):
            scene = Scene()
            scene.index = idx
            scene.image_prompt = scene_data.get("image_prompt", "")
            scene.audio_text = scene_data.get("audio_text", "")
            scenes.append(scene)
        

        project = Project()
        project.name = "fear at night"
        project.story = "A boy sleeping on his bed..."
        project.scenes = scenes
        
        # Optional: Print or return the results
        print(f"Created project with {len(scenes)} scenes:")
        for scene in scenes:
            print(f"Scene {scene.index}:")
            print(f"Image prompt: {scene.image_prompt}")
            print(f"Audio text: {scene.audio_text}\n")
        
        return project
        
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
    except Exception as e:
        print(f"Error processing response: {e}")
    # print(response)
    # format_res = response['text'].strip()
    # if format_res.startswith("json") and format_res.endswith(""):
    #     format_res = format_res[7:-3].strip()
    # try:
    #     response_json = json.loads(format_res)
    #     print("Formatted res JSON - ", response_json)
    # except json.JSONDecodeError as e:
    #             print(f"JSON Decode Error: {e}")


create_scenes(prompt_template)

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



