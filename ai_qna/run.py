import os
import sys
from typing import List
from .extract_repo import extract_local_directory
from .qna import AskBot
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich import print as rprint


def get_repo_info(directory_path: str) -> dict:
    """Extract basic information about the repository."""
    repo_name = os.path.basename(directory_path)
    file_count = sum(len(files) for _, _, files in os.walk(directory_path))
    return {"name": repo_name, "path": directory_path, "file_count": file_count}


def generate_system_message(repo_info: dict) -> str:
    """Generate a system message for the AI to set the context."""
    return f"""You are an AI assistant specializing in explaining code repositories. You're currently analyzing the '{repo_info['name']}' repository located at '{repo_info['path']}', which contains approximately {repo_info['file_count']} files.

Your task is to answer questions about this specific repository based on the code and documentation provided. Please keep the following guidelines in mind:

1. Focus solely on the information present in the given repository.
2. If you're unsure or if the information isn't available in the repository, say so.
3. Provide concise, technically accurate answers tailored to a developer audience.
4. Use simple language and explain complex concepts when necessary.
5. If relevant, suggest where in the codebase the user might look for more information.

Remember, you're here to help developers understand this specific codebase better."""


def query_repo(directory_path: str):
    """Main function to handle the QnA process with an improved user experience."""
    console = Console()

    with console.status("Preparing the QnA environment...", spinner="dots"):
        text_file_path = extract_local_directory(directory_path)
        repo_info = get_repo_info(directory_path)
        system_message = generate_system_message(repo_info)
        bot = AskBot(text_file_path, system_message)

    console.print(
        Panel.fit(
            f"[bold green]Welcome to the {repo_info['name']} Repository Q&A Assistant![/bold green]\n"
            f"Ask questions about the codebase, and I'll do my best to answer based on the repository content.\n"
            "Type 'exit' to quit the program."
        )
    )

    while True:
        query = Prompt.ask(
            "\n[bold cyan]What would you like to know about the repository?[/bold cyan]"
        )

        if query.lower() == "exit":
            console.print(
                "[yellow]Thank you for using the Repository Q&A Assistant. Goodbye![/yellow]"
            )
            break

        with console.status(
            "Analyzing the repository and generating an answer...", spinner="dots"
        ):
            response = bot.query(query)

        rprint(Panel(response["result"], title="Answer", border_style="green"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run.py <local repository directory>")
        sys.exit(1)

    directory_path = sys.argv[1]
    query_repo(directory_path)
