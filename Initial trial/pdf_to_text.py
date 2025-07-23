import os
import fitz

def convert_pdfs_to_text(pdf_folder,text_folder):
    if not os.path.exists(text_folder):
        os.makedirs(text_folder)

    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            file_path=os.path.join(pdf_folder,file_name)
            text_file_name=os.path.splitext(file_name)[0]+".txt"
            text_file_path=os.path.join(text_folder,text_file_name)
            with fitz.open(file_path) as doc:
                text = ""
                for page in doc :
                    text += page.get_text()
            with open (text_file_path , "w", encoding ="utf-8") as text_file :
                text_file.write (text)
            print (f"Converted {file_name} to {text_file_name}")

if __name__ == "__main__":
    pdf_folder = "Data"
    text_folder = "DataTxt"
    convert_pdfs_to_text (pdf_folder,text_folder)