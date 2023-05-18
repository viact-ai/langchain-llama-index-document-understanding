def read_csv_as_str(filepath: str) -> str: 
    with open(filepath,"r", encoding="utf-8") as f: 
        data = f.readlines()    
    str_data = ""
    for row in data: 
        str_data += row
    return str_data