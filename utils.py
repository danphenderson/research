def write_text(path, text):
    """Write text to a file.

    Args:
        path (str): The file path where the text will be written.
        text (str): The text content to write to the file.
    """
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)
