import streamlit as st
from dotenv import load_dotenv

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
OPENAI_API_VERSION = '2023-12-01-preview'

import sys
import io
import logging
from typing import List, Dict
import json
from uuid import uuid4 as uuid
from dataclasses import dataclass
from openai import OpenAI
import time
from langchain_openai import ChatOpenAI
import re

import pandas as pd
from duckduckgo_search import DDGS
from langgraph.pregel import GraphRecursionError
from typing import Tuple, Dict


model = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-4o-mini"
)

def llm(query):
  # Add retry logic with exponential backoff
  max_retries = 50  # Maximum number of retries
  retry_delay = 1  # Initial retry delay in seconds

  for i in range(max_retries):
    try:
      response = model.invoke(query)
      return response.content
    except openai.error.RateLimitError as e:
      if i < max_retries - 1:
        print(f"Rate limit error, retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)
        retry_delay *= 2  # Exponential backoff
      else:
        raise e  # Raise the exception if max retries reached


@dataclass
class Query:
    """
    A dataclass to store the query name and its content.
    """
    name: str
    content: str

    def __repr__(self):
        return f"Query name: {self.name}\n" \
                f"Query content: {self.content}\n"

@dataclass
class Resource:
    """
    A dataclass to store the resource name, description and its content.
    """
    name: str
    description: str
    content: str = None

    def set_content(self, content):
        self.content = content

    def get_content(self):
        return (self.name, self.description, self.content)

    def __repr__(self):
        return f"Resource name: {self.name}\n" \
                f"Resource description: {self.description}\n" \
                f"Resource content: {self.content}\n"
    

class QueryParser:
  def __init__(self, input_data):
        self.input_data = input_data

  def parse_query_name(self):
        for key in ('query-name', 'query_name'):
            if key in self.input_data:
                return self.input_data[key]
        raise KeyError("Could not find the query name")
  
  def parse_query_content(self, query_path):
        # If the user provided "query_content", skip reading from disk
        if "query_content" in self.input_data and self.input_data["query_content"] is not None:
            return self.input_data["query_content"]

        # Otherwise, read the file from disk
        with open(query_path, "r", encoding="utf-8") as f:
            return f.read()
        
  def parse_query(self):
        query_name = self.parse_query_name()
        query_content = self.parse_query_content(query_name)
        return Query(query_name, query_content)


  # def parse_query_name(self, input_data):
  #   # Parse query name. Appears in the exercise in two different ways.
  #   for key in ('query-name', 'query_name'):
  #     if key in input_data:
  #       return input_data[key]
  #   raise KeyError("Could not find the query name")

  # def parse_query_content(self, query_path):
  #   with open(query_path) as f:
  #     return f.read()

  # def parse_query(self, input_data):
  #   query_name = self.parse_query_name(input_data)
  #   query_content = self.parse_query_content(query_name)
  #   return Query(query_name, query_content)


class ResourcesParser:

  def parse_resource(self, resource_data) -> List[Resource]:
    # Parse resources. Appears in the exercise in two different ways.
    resources = []  # List of dicts of resources
    for key in ('name', 'file_name'):
      if key in resource_data:
        name = resource_data[key]
        description = resource_data['description']
        content = resource_data.get("content", None)
        res = Resource(name, description)
        if content is not None:
          res.set_content(content)

        return res
        # return Resource(name, description)
    raise KeyError("Could not find the name of the resource")

  def load_resources_contents(self, resources):
    import os
    for res in resources:
      if res.content is not None:
        continue
       
      if os.path.exists(res.name):
         with open(res.name, "r", encoding="utf-8") as f:
            if res.name.endswith('csv'):
              df = pd.read_csv(f)
              # Append the columns index to the resource description
              columns_index = json.dumps({column: i for i, column in enumerate(df.columns)})
              res.description += columns_index   
              # Go back to the start of the file to read the raw CSV text for res.content
              f.seek(0)
              csv_text = f.read()
              res.set_content(csv_text)
            else:
              res.set_content(f.read())
      else:
         res.set_content("")
                 

    # for res in resources:
    #   with open(res.name) as f:
    #     if res.name.endswith('csv'):
    #       df = pd.read_csv(f)
    #       res.set_content(f)
    #       columns_index = json.dumps({column: i for i, column in enumerate(list(df.columns))})
    #       res.description += columns_index
    #     else:
    #       res.set_content(f.read())

  def parse_resources(self, input_data):
    resources = [self.parse_resource(res) for res in input_data['file_resources']]
    self.load_resources_contents(resources)
    return resources


class InputParser:

  def __init__(self, input_data):
    self.input_data = input_data
    # self.query_parser = QueryParser()
    self.query_parser = QueryParser(input_data)
    self.resources_parser = ResourcesParser()

  def parse_query(self):
    # return self.query_parser.parse_query(self.input_data)
    return self.query_parser.parse_query()
  

  def parse_resources(self):
    return self.resources_parser.parse_resources(self.input_data)

  def parse_input(self):
    return (
        self.parse_query(),
        self.parse_resources()
        )


class Loader:
  def __init__(self, user_input):
    input_parser = InputParser(user_input)
    self.query, self.resource = input_parser.parse_input()

  def get_query(self):
    return self.query

  def get_resources(self):
    return self.resource


def load_input(path):
  with open(path) as f:
    data = json.load(f)
  return data

def log(state, msg):
  state['log'].append(msg)
  print(msg)
  return state

def log_param(state, param_names, param_values):
  for name, value in zip(param_names, param_values):
    if isinstance(value, str):
      value = value[:50]
    log(state, f"Parameter {name} = {value}")

def initialize(state):
    name = "initialize"
    log(state, f'**Entering agent {name}**')

    user_input = load_input('input.json')
    state['query_name'] = user_input['query_name']
    loader = Loader(user_input)
    query = loader.get_query()
    resources = loader.get_resources()
    state['query'] = query.content
    state['resources'] = [res.get_content() for res in resources]
    
    log(state, f'**Leaving agent {name}**')
    return state
  
def plan(state):
    name = "plan"
    log(state, f'**Entering agent {name}**')

    # Terminate if exceeded the maximum LLM/function calls
    if state['llm_invocations'] >= 10:
      state['error'] = "Exceeded maximum number of LLM calls"
      state['next_step'] = "Finalize"
      state['next_step_args'] = {}
      return state
    if state['func_invocation'] >= 10:
      state['error'] = "Exceeded maximum number of function calls"
      state['next_step'] = "Finalize"
      state['next_step_args'] = {}
      return state

    # Create prompt template
    prompt_template = """
    Help us decide what should be the next action.

    The possible actions are:

    1. GeneratePythonCode: This function takes an analysis request,
    and a csv input_file.

    2. NamedEntityRecognition: The function is given a text file and an "entity type" (could be
    "student", "hobby", "city" or any type of entity). It returns a
    string representing a list of entities of that type in the file

    3. FileWriter: Writes to a text file

    4. NER: Named Entity Recognition tool

    5. WebSearch: search the web for information

    6. Finazlize: This function completes our run since we have an answer to the query.

    Our overall task is: {query}

    We have the following resources available to us: {resources}

    The steps we did so far:
    {past_steps}
    """

    # Formatting previous steps
    if len(state['past_steps']) == 0:
      past_steps = "None"
    elif len(state['past_steps']) == 1:
      past_steps = state['past_steps'][0]
    else:
      past_steps = "\n".join(state['past_steps'])

    # Create prompt
    resources_repr = "\n".join([f"'Name': {res[0]}, 'description': {res[1]}" for res in state['resources']])
    prompt = prompt_template.format(
        query=state['query'],
        resources=resources_repr,
        past_steps=past_steps
    )
    examples = """"
    Your response should be one of the following:
    {"tool": "GeneratePythonCode", "args": {"analysis_request": "<analysis request>", "input_file": "<input_file>", "columns": "<columns>", "row_example": "<row_example>", "python_file": "<python_filename>", "output_format": "<output_format>"}
    {"tool": "ExecutePython", "args": {"program_fn": "<input_program_name>", "output_file": "<output_filename>"}}
    {"tool": "NamedEntityRecognition", "args": {"file_name": "<file_name>", "entity_type": "<entity_type>"}}
    {"tool": "FileWriter", "args": {"file_name": "<file_name>", "file_content": "<file_content_string>"}}
    {"tool": "WebSearch", "args": {"a_entity": "<a_entity>", "a_attribute": "<a_attribute>"}}
    {"tool": "Finalize", "args": {}}
    """
    prompt += examples

    # Query GPT + parse
    response = llm(prompt)
    state['llm_invocations'] += 1

    # parsed_response = json.loads(response)
    # Post-process response to remove possible code fences
    # Strip whitespace
    # Clean and extract JSON
    response = response.strip().replace("```json", "").replace("```", "").strip("`").strip()
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    # Remove surrounding triple backticks and 'json' hint if present
    # response = response.replace("```json", "").replace("```", "").strip("`").strip()

    # Check if the response is valid JSON before parsing
    if json_match:
        json_content = json_match.group(0)
        try:
            parsed_response = json.loads(json_content)
        except json.JSONDecodeError as e:
            print(f"Error parsing extracted JSON: {e}")
            print(f"Extracted content: {json_content}")
            state['next_step'] = "Finalize"
            state['next_step_args'] = {}
            return state
    else:
        print("No JSON object found in the response")
        print(f"Response content before parsing: {response}")
        state['next_step'] = "Finalize"
        state['next_step_args'] = {}
        return state


    # Update state
    state['next_step'] = parsed_response['tool']
    state['next_step_args'] = parsed_response['args']
    log(state, f'**Leaving agent {name}**')
    return state


def plan_helper(state):
    return state['next_step']

def clean_code(code_str: str) -> str:
    """
    Extract the code content from a triple backtick block if present.
    Otherwise return the original text.
    """
    # Attempt to find a code block of the form ```python ... ```
    match = re.search(r"```(?:python)?(.*?)```", code_str, re.DOTALL)
    if match:
        # Group(1) is the text between the backticks
        extracted_code = match.group(1).strip()
        return extracted_code
    else:
        # If no fenced block, just return the original text
        return code_str.strip()
    

def generate_analysis_program(state) -> str:
    name = "generate_analysis_program"
    log(state, f'**Entering agent {name}**')
    state['func_invocation'] += 1

    # Parse args from state
    args = state['next_step_args']
    analysis_request = args['analysis_request']
    input_file = args['input_file']
    columns = args['columns']
    row_example = args['row_example']
    output_file = args['python_file']
    output_format = args['output_format']

    # Log parameters
    params = ["analysis_request", "input_file", "columns", "row_example", "output_file", "output_format"]
    params_values = [analysis_request, input_file, columns, row_example, output_file, output_format]
    log_param(state, params, params_values)

    # Create a prompt template
    prompt_template = """
    I want you to write me a Python script that analyses data from a file.

    The file I want the script to analyze is called {input_file}.
    It is a `.csv` file with the following columns: {columns}.
    For example, the row of data (after the column names row): {row_example}.

    Here is the instructions for the analysis itself.
    Your Python script should reflect these requirements:
    {analysis_request}
    The desired output format is {output_format}.
    The last line of your code should be printing the results

    The Python script should be saved in {output_file}
    """

    # Create a prompt
    prompt = prompt_template.format(
        analysis_request=analysis_request,
        input_file=input_file,
        columns=columns,
        row_example=row_example,
        output_file=output_file,
        output_format=output_format
    )

    # Query GPT
    response = llm(prompt)

    code = clean_code(response)
    # Save Python file
    with open(output_file, 'w') as f:
      f.write(code)

    if "created_files" not in state:
        state["created_files"] = []
    state["created_files"].append(output_file)
  
    # Verify saved file content
    with open(output_file, 'r') as f:
        saved_code = f.read()

    step = f"""
    Created a python program that analyzes {input_file}.
    The program does {analysis_request}
    Output of this program is saved in {output_file}
    """
    state['past_steps'].append(step)

    log(state, f'**Leaving agent {name}**')
    return state
def execute_Python_program(state) -> str:
    name = "execute_Python_program"
    log(state, f'**Entering agent {name}**')
    state['func_invocation'] += 1

    # Define success and failure msgs
    SUCCESS_MSG = "Program executed successfully"
    FAILURE_MSG = "Program failed to load or execute"

    # Unpacking state
    args = state['next_step_args']
    program_fn = args['program_fn']
    output_fn = args['output_file']

    # Logging params
    log_param(state, ['program_fn', 'output_fn'], [program_fn, output_fn])

    # Redirect stdout to capture the output
    sys_stdout = sys.stdout

    # Read the file from disc, execute
    try:
      with open(program_fn, 'r') as f:  # Assume writing was successful
        code = f.read()

      redirected_output = sys.stdout = io.StringIO()
      exec(code, {"__name__": "__main__"})

      results = redirected_output.getvalue()
      sys.stdout = sys_stdout

      # Store it in the pipeline state
      state['program_output'] = results
  
      log(state, SUCCESS_MSG)

      with open(output_fn, 'w') as f:  # We are requested to implement a function that writes to a file. Should probably use that one instaeed of do it here.
        f.write(results)

      if "created_files" not in state:
        state["created_files"] = []
      state["created_files"].append(output_fn)

      state['past_steps'].append(f"Executed {state['next_step_args']['program_fn']}")
      log(state, f'**Leaving agent {name}**')
      return state

    except Exception as e:
      sys.stdout = sys_stdout
      state['past_steps'].append(f"Executed {state['next_step_args']['program_fn']}, but encountered an error. The script is corrupt.")
      log(state, f"{FAILURE_MSG}: {e}")
      log(state, f'**Leaving agent {name}**')
      return state
    
def extract_entities_from_file(state) -> str:
    name = "extract_entities_from_file"
    log(state, f'**Entering agent {name}**')
    state['func_invocation'] += 1

    # Extracting arguments
    file_name = state['next_step_args']['file_name']
    entity_type = state['next_step_args']['entity_type']
    log_param(state, ['file_name', 'entity_type'], [file_name, entity_type])

    # Prompt template
    prompt_template = """
    You are given a text, and we want to extract entities of type {entity_type} from it.

    The text:
    {text}
    """

    with open(file_name) as f:
      text = f.read()

    # Prompt
    prompt = prompt_template.format(entity_type=entity_type, text=text)

    # Query GPT
    response = llm(prompt)

    state['past_steps'].append(f"Extracted entity type '{entity_type}' from file {file_name}: {response}")
    log(state, f'**Leaving agent {name}**')
    return state


def write_file(state) -> str:
    name = "write_file"
    log(state, f'**Entering agent {name}**')
    state['func_invocation'] += 1

    file_name = state['next_step_args']['file_name']
    file_content = state['next_step_args']['file_content']
    log_param(state, ['file_name', 'file_content'], [file_name, file_content])

    with open(file_name, 'w') as f:
      f.write(file_content)

    if "created_files" not in state:
        state["created_files"] = []
    state["created_files"].append(file_name)

    state['past_steps'].append(f"Wrote file {file_name}")
    log(state, f'**Leaving agent {name}**')
    return state

def Internet_search_attribute(state) -> str:
    name = "Internet_search_attribute"
    log(state, f'**Entering agent {name}**')
    state['func_invocation'] += 1

    a_entity = state['next_step_args']['a_entity']
    a_attribute = state['next_step_args']['a_attribute']
    log_param(state, ['a_entity', 'a_attribute'], [a_entity, a_attribute])

    query = f"{a_entity} {a_attribute}"
    time.sleep(1)  # Wait for 1 second before making the request
    results = DDGS().text(query, max_results=5,backend="html")
    answer = results[0]['body'] if results else "No results found"

    response = {"a_entity": a_entity, a_attribute: answer}
    state['past_steps'].append(f"Searched the internet for {query} and got {response}")
    log(state, f'**Leaving agent {name}**')
    return state


def finalize(state):
    name = "Finalize"
    log(state, f'**Entering agent {name}**')

    query_name = state['query_name']
    log_filename = f"log_{query_name}"

    log_string = "\n".join(state['log'])
    if state['error']:
      log_string += f"\nExecution interrupted: {state['error']}"

    with open(log_filename, 'w') as f:
      f.write(log_string)

    log(state, f'**Leaving agent {name}**')
    return state

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END


class State(TypedDict):
  llm_invocations: int
  func_invocation: int
  query_name: str
  query: str
  resources: List[Tuple[str, str, pd.DataFrame | str]]
  log: List[str]
  next_step: str
  next_step_args: Dict
  past_steps: List[str]
  error: str | None
  created_files: List[str]

# Initiating the Graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("Initialize", initialize)
workflow.add_node("Plan", plan)
workflow.add_node("GeneratePythonCode", generate_analysis_program)
workflow.add_node("ExecutePython", execute_Python_program)
workflow.add_node("NamedEntityRecognition", extract_entities_from_file)
workflow.add_node("FileWriter", write_file)
workflow.add_node("WebSearch", Internet_search_attribute)
workflow.add_node("Finalize", finalize)

# Add edges
workflow.set_entry_point("Initialize")
workflow.add_edge("Initialize", "Plan")
workflow.add_edge("GeneratePythonCode", "Plan")
workflow.add_edge("ExecutePython", "Plan")
workflow.add_edge("FileWriter", "Plan")
workflow.add_edge("NamedEntityRecognition", "Plan")
workflow.add_edge("WebSearch", "Plan")
workflow.add_conditional_edges("Plan", plan_helper)
workflow.add_edge("Finalize", END)

# Finalize building graph
runnable = workflow.compile()

past_steps = []
log_list = []
state = State(query="", llm_invocations=0, func_invocation=0, log=log_list, past_steps=past_steps, error=None)


def run_pipeline(input_data: dict):
    """
    1) Write the user's input_data to 'input.json'
    2) Create a fresh state
    3) Invoke the compiled workflow
    4) Return the final 'state'
    """
    # Step 1: Save input_data to a local file named 'input.json'
    import json
    with open("input.json", "w") as f:
        json.dump(input_data, f, indent=2)

    # Step 2: Create fresh lists for logs and past steps
    past_steps = []
    log_list = []
    new_state = State(
        query="",
        llm_invocations=0,
        func_invocation=0,
        log=log_list,
        past_steps=past_steps,
        error=None
    )

    # Step 3: Actually run the workflow
    final_state = runnable.invoke(new_state)

    # Step 4: Return whatever you want the user to see
    return final_state
