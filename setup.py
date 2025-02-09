from setuptools import setup, find_packages

setup(
    name="vyper-ai",
    version="1.0.0",
    description="Dynamic AI Team Orchestration System",
    author="Vyper AI Team",
    packages=find_packages(),
    install_requires=[
        "python-dotenv>=1.0.0",
        "rich>=13.5.0",
        "pydantic>=2.0.0",
        "PyYAML>=6.0.1",
        "aiohttp>=3.8.5",