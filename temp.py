import asyncio
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

from config import env
from storage import Storage
import json
import datetime


prompt_template = PromptTemplate(
    input_variables=["current_time", "history", "question"],
    template="""You are a Swiggles Hospital AI Assistant. Your role is to assist patients with booking appointments,
                retrieving medical reports, and reporting emergencies. Always respond in a professional and empathetic manner.
                If the request is unclear, ask for clarification.

                If the user asks a general question about the hospital, respond appropriately without invoking any tools.

                If the user requests to book an appointment, retrieve a medical report, or report an emergency,
                ensure that all necessary information is provided. If any required information is missing, ask the user for it.

                Your response should be in JSON format with the following fields:
                - 'tool': The tool to be invoked (e.g., 'book_appointment', 'get_report', 'report_emergency', or 'general' if no tool is needed).
                - 'input': Input required for the tool (if applicable).
                - 'output': The response to the user.
                - 'missing_info': Any missing information required to complete the request (if applicable).
                
                Information required for each tool:
                - 'book_appointment': 'time'
                - 'get_report': None
                - 'report_emergency': None

                Example 1 JSON response:
                {{
                    'tool': 'book_appointment',
                    'input': 'null',
                    'output': 'Please provide the time for the appointment.',
                    'missing_info': 'time'
                }}
                
                Example 2 JSON response:
                {{
                    'tool': 'book_appointment',
                    'input': '10:00 AM {{with a proper date and day in string format understandable by the human}}',
                    'output': 'Appointment booked successfully for 10:00 AM.'
                    'missing_info': 'null'
                }}
                

                Ensure the response is valid JSON and can be parsed by Python's json.loads() function.
                Do not include any additional text outside the JSON object.
                # give current time and date with day to ai so that if user asks for appointment it can book it for the user
                If the user want to book an appointment, the user will provide the time of the appointment, to pass input to tool, here is the current time with day in 24h format {current_time}

                Conversation History Between AI and user: {history}
                Question: {question}
                Answer:""")

llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=env["GOOGLE_API_KEY"],
    temperature=0.5,
)


class AI:
    def _init_(self):
        self.chain = LLMChain(llm=llm, prompt=prompt_template)
        self.storage = Storage()

    async def interact(self, patient_id: str, message: str) -> str:
        try:
            history = self.storage.get_past_messages(
                patient_id) or "No past messages found for this patient"
            self.storage.add_to_conversation(patient_id, "user", message)

            response = self.chain.invoke({
                "current_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " " + datetime.datetime.now().strftime("%A"),
                "history": history,
                "question": message
            })

            format_res = response['text'].strip()
            if format_res.startswith("json") and format_res.endswith(""):
                format_res = format_res[7:-3].strip()

            try:
                response_json = json.loads(format_res)
                print("Formatted res JSON - ", response_json)
                tool = response_json.get("tool")
                output = response_json.get("output")
                missing_info = response_json.get("missing_info", "")

                if tool == "book_appointment":
                    if missing_info == "time":
                        self.storage.add_to_conversation(
                            patient_id, "bot", output)
                        return output
                    else:
                        input = response_json.get("input")
                        result = self.book_appointment(patient_id, input)
                        self.storage.add_to_conversation(
                            patient_id, "bot", result)
                        return result

                elif tool == "get_report":
                    result = self.get_report(patient_id)
                    self.storage.add_to_conversation(patient_id, "bot", result)
                    return result

                elif tool == "report_emergency":
                    result = self.report_emergency(patient_id)
                    self.storage.add_to_conversation(patient_id, "bot", result)
                    return result

                else:
                    self.storage.add_to_conversation(patient_id, "bot", output)
                    return output

            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {e}")
                self.storage.add_to_conversation(
                    patient_id, "bot", "An error occurred while processing your request.")
                return "An error occurred while processing your request."

        except Exception as e:
            print(f"Error @ai.py.interact: {e}")
            return f"An error occurred @ai.py.interact: {e}"

    def book_appointment(self, patient_id: str, input: str) -> str:
        return self.storage.add_to_appointments(patient_id, input)

    def get_report(self, patient_id: str) -> str:
        return f"Medical report for {patient_id}: Blood pressure - 120/80, Cholesterol - 190 mg/dL."

    def report_emergency(self, patient_id: str) -> str:
        return self.storage.add_to_emergencies(patient_id)


async def test():
    ai = AI()
    # res = await ai.interact("6969", "Who are you")
    # print(res)
    res = await ai.interact("6969", "I am getting stomach pain, can you book an appointment for 10am tomorrow")
    print(res)
    # res = await ai.interact("6969", "I have an emergency, help me!")
    # print(res)
    # res = await ai.interact("6969", "Can I get my medical reports")
    # print(res)

if _name_ == "_main_":
    asyncio.run(test())





from typing import List
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from dotenv import load_dotenv
import os
import json

load_dotenv()

llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.5,
)

prompt_template = PromptTemplate(
    input_variables=["name", "story"],
    template="""
                
                You are a professional story writer for short form video content where the video will be created with a minimum of 10 scenes. You write the scenes for the story given by the user. The video that will be generated will comprise of images with animations and audio telling the story.

                The scene will be an array which will consist of image_prompt and audio_text of all the scenes like a json file of each scene.
                You need to write the audio_text of the scene in such a way that it will be a continuation of the previous scene audio_text so that the story will be continuous. The images need not be continous but should atleast be a little related to the scene.

                The scene will consist of a image with small animations like zoom in, zoom out, pan, etc. The audio will be a voice over of the text of the scene.
                The image that will be displayed will be generated by ai with the image_prompt from the scene. The audio will be generated from the audio_text from the scene.

                Each scene should be too long and each scene will be of the length of the audio_text. That is you need to generate the audio_text of each scene with 10-15 words and not more, so structure the scenes accordingly.

                You as the scene writer for the story needs to write a description for the image to be generated by ai and also you need to wite short audio overlays for the scene being delivered at the moment.
                So you need to think about the how many scenes should be there and how to structure them properly and how you need to write the image_prompt for the scene and how you need to write the audio_text for the scene.

                The output should be a json object with the scenes array which will consist of image_prompt and audio_text of all the scenes.
                
                Ensure the response is valid JSON and can be parsed by Python's json.loads() function.
                Do not include any additional text outside the JSON object.
                

                name of the story given by the user: {name}
                story given by the user: {story}
                Answer:""")




class Scene:
    index: int
    image_prompt: str
    audio_text: str
    image_file: str | None
    audio_file: str | None


class Project:
    name: str
    story: str
    scenes: List[Scene]


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

def create_images():
    pass


def create_audio():
    pass


def image_to_video_with_captions_overlay_and_audio():
    pass


def merge_video_scenes():
    pass


def main():
    pass


'''
    get story name and story from user
    create scenes using ai
    create images using ai
    create audio using gtts
    merge images, audio and captions to single scene 
    merge all scenes to single video
'''

if _name_ == "_main_":
    main()