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