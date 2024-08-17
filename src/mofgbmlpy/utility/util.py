def dash_case_to_class_name(txt):
    """Convert a text in dash case format to a class name format. e.g. an-example becomes AnExample

    Args:
        txt (str): text to be converted

    Returns:
        str: New text
    """
    parts = txt.split('-')
    parts = [p.capitalize() for p in parts]
    return ''.join(parts)


def dash_case_to_snake_case(txt):
    """Convert a text in dash case format to snake case. e.g. an-example becomes an_example

    Args:
        txt (str): text to be converted

    Returns:
        str: New text
    """
    return txt.replace('-', '_')


def snake_case_to_dash_case(txt):
    """Convert a text in snake case to dash case format. e.g. an_example becomes an-example

    Args:
        txt (str): text to be converted

    Returns:
        str: New text
    """
    return txt.replace('_', '-')
