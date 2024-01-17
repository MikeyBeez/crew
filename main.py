from crewai import Agent, Task, Crew, Process
import os

from langchain.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()
from langchain.llms import Ollama
ollama_openhermes = Ollama(model="openhermes")
ollama_llama2 = Ollama(model="llama2")

researcher = Agent(role='Researcher', goal='Research methods to sell this property <address>', backstory='You are an AI research assistant', tools=[search_tool], verbose=True, llm=ollama_openhermes, allow_delegation=False)

writer = Agent(role='Writer', goal='Write compelling and engaging reasons to market this property', backstory='You are an AI master mind capable of marjeting any real estate', verbose=True, llm=ollama_llama2, allow_delegation=False)


task1 = Task(description='Investigate <address>', agent=researcher)
task2 = Task(description='Investigate sure fire ways to market this property', agent=researcher)

task3 = Task(description='Write a list of tasks to market this property', agent=writer)


crew = Crew(agents=[researcher, writer], tasks = [task1,task2,task3], verbose=2, process=Process.sequential)

result = crew.kickoff()

print(result)
