from __future__ import unicode_literals
from prompt_toolkit import PromptSession, print_formatted_text, HTML
from prompt_toolkit.shortcuts import ProgressBar
from prompt_toolkit.styles import Style

import warnings

from fastai.text import load_learner

my_style = Style.from_dict(
    {
        "system": "#1FB7F6",
        "like": "#45F939",
        "dislike": "#F43911",
        "continue": "#8B9FA7",
    }
)


def RNN_prediction(classifier, input_txt):
    # Supress tensorflow warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sentiment = int(classifier.predict(input_txt)[0])

    if sentiment == 1:
        print_formatted_text(
            HTML("<like>Thank you for liking us!\n</like>"), style=my_style
        )
    elif sentiment == 0:
        print_formatted_text(
            HTML("<dislike>Whoops, it seems that you don't like the food.\n</dislike>"),
            style=my_style,
        )
    else:
        print_formatted_text("whoops")

    return sentiment


print_formatted_text(
    HTML("<system>Initiating food review robot...</system>"), style=my_style
)
with ProgressBar() as pb:
    classifier = load_learner("./models")
print_formatted_text(
    HTML(
        "<system>Robot loaded. Please enter your first review. Press Control+D to exit.</system>"
    ),
    style=my_style,
)
print("\n")


def main():
    session = PromptSession()

    while True:
        try:
            text = session.prompt("> ")
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        else:
            RNN_prediction(classifier, text)
            print_formatted_text(
                HTML("<continue>Please enter your next review...</continue>"),
                style=my_style,
            )

    print_formatted_text(HTML("<system>See you next time!</system>"), style=my_style)


if __name__ == "__main__":
    main()
