import os
import sys
from .extract_repo import extract_local_directory
from .qna import AskBot


def query_repo(directory_path: str):
    """"""
    text_file_path = extract_local_directory(directory_path)
    bot = AskBot(text_file_path)
    repo_name = str(os.path.splitext(os.path.basename(text_file_path))[0]).replace(
        "_code", ""
    )

    exit = False
    while not exit:
        query = input(f"What do you wish to know about {repo_name}?\n")

        if str(query).upper() == "EXIT":
            print("Exiting...")
            exit = True
            sys.exit(1)

        response = bot.query(query)
        print(response["result"])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run.py <local repository directory>")
        sys.exit(1)

    directory_path = sys.argv[1]
    query_repo(directory_path)
