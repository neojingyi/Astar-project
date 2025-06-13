import re

def chunk_by_gri_headings(text):
    pattern=r'(GRI\d{3}.*?\n)'
    parts = re.split(pattern,text)
    result=[]
    if len(parts)<=1:
        paras=text.split('\n\n')
        return [p.strip() for p in paras if len(p.strip())>50]
    for i in range(1,len(parts),2):
        heading=parts[i].strip()
        body=parts[i+1].strip()
        result.extend(f"{heading}\n{body}")
    return result