from crewai import Agent, Task, Crew, Process
import os

from langchain.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()
from langchain.llms import Ollama
ollama_openhermes = Ollama(model="openhermes")
ollama_llama2 = Ollama(model="llama2")

researcher = Agent(role='Researcher', 
                   goal='Research methods to create an agent that runs on ollama, uses openhermes, uses memgpt, use both typing or stt/tts, is always running and accessible, learns.  Write the python code.',
                   backstory='You are an AI research assistant',
                   tools=[search_tool],
                   verbose=True,
                   llm=ollama_openhermes,
                   allow_delegation=False)

writer = Agent(role='Writer',
               goal='Write complete documentation along with code',
               backstory='You are an AI master mind capable of completing python projects',
               verbose=True,
               llm=ollama_llama2,
               allow_delegation=False)


task1 = Task(description='Investigate methods and write code', agent=researcher)
task2 = Task(description='write accompanting documentation', agent=researcher)

task3 = Task(description='integrate code and documentation', agent=writer)


crew = Crew(agents=[researcher, writer], tasks = [task1,task2,task3], verbose=2, process=Process.sequential)

result = crew.kickoff()

print(result)
