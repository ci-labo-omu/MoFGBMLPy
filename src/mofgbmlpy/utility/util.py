def dash_case_to_class_name(txt):
    parts = txt.split('-')
    parts = [p.capitalize() for p in parts]
    return ''.join(parts)


def dash_case_to_snake_case(txt):
    return txt.replace('-', '_')
