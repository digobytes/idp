import streamlit as st
import pandas as pd
import boto3
from io import BytesIO
import boto3
from botocore.exceptions import ClientError
import json  # Import the json module
import os    

st.set_page_config(page_title="AI Underwriter - POC")
st.title("AI Underwriter - POC")

st.sidebar.title("Upload the document to user folder to process")

# connect to boto3 and initiate services

comprehend_medical = boto3.client(service_name='comprehendmedical', region_name='us-east-1')
textract = boto3.client('textract', region_name='us-east-1') 
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')  
model_id = "amazon.titan-text-premier-v1:0"  # Example, replace with the correct model ID
s3_bucket_name = 'aps-aws-input'
s3 = boto3.resource('s3')

# Function to upload file to S3
def upload_file_to_s3(file, folder_name, file_type):

    # Append file type to the uploaded file name
    new_file_name = f"{file.name.split('.')[0]}_append_{file_type}.{file.name.split('.')[-1]}"

    # Create folder path with file name appended with type
    folder_path = f"{folder_name}/{new_file_name}"
    
    # Save file to S3 with metadata
    file_metadata = {
        'Content-Type': file.type,
        'File-Type': file_type
    }
    s3.Bucket(s3_bucket_name).put_object(Key=folder_path, Body=file, Metadata=file_metadata)
    st.write(f"File '{new_file_name}' uploaded to folder '{folder_name}' as {file_type}.")

# Get list of existing folders in the S3 bucket
def get_existing_folders(bucket_name):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket_name, Delimiter='/')
    folders = [prefix['Prefix'].split('/')[0] for prefix in response.get('CommonPrefixes', [])]
    return folders

# Function to list all files in a folder in S3
def list_files_in_folder(bucket_name, folder_name):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name + "/", Delimiter="/")

    file_list = []
    # If there are objects (files) under the folder
    if 'Contents' in response:
        for obj in response['Contents']:
            file_list.append(obj['Key'])
    return file_list

# Function to check if a document needs medical comprehension
def needs_medical_comprehension(document_type):
    medical_documents = ['MedicalReport', 'APS']
    return document_type in medical_documents

# Function to extract text from a document using Textract
def extract_text_from_textract(s3_bucket_name, file_key, form_type):
    textract = boto3.client('textract')
    text = ""
    
    if form_type == "Form document extraction":
        response = textract.analyze_document(Document={'S3Object': {'Bucket': s3_bucket_name, 'Name': file_key}}, FeatureTypes=["TABLES", "FORMS", "SIGNATURES", "LAYOUT"])
        for item in response["Blocks"]:
            if item["BlockType"] == "LINE":
                text += item["Text"] + "\n"
    else:
        response = textract.detect_document_text(Document={'S3Object': {'Bucket': s3_bucket_name, 'Name': file_key}})
        for item in response["Blocks"]:
            if item["BlockType"] == "LINE":
                text += item["Text"] + "\n"
    return text

# Function to extract medical entities from text using Comprehend Medical
def extract_medical_entities(text):
    comprehend_medical = boto3.client('comprehendmedical')
    response = comprehend_medical.detect_entities_v2(Text=text)
    entities = response['Entities']
    
    entity_data = []
    for entity in entities:
        entity_data.append({
            'Text': entity.get('Text', ''),
            'Category': entity.get('Category', ''),
            'Type': entity.get('Type', ''),
            'Traits': [trait['Name'] for trait in entity.get('Traits', [])]
        })
    
    return pd.DataFrame(entity_data)

# Function to map conditions to ICD-10 codes using Comprehend Medical
def map_conditions_to_icd10(text):
    comprehend_medical = boto3.client('comprehendmedical')
    icd_response = comprehend_medical.infer_icd10_cm(Text=text)
    icd_entities = icd_response['Entities']
    
    icd_data = []
    for condition in icd_entities:
        icd_data.append({
            'Condition': condition.get('Text', ''),
            'ICD-10 Code': condition['ICD10CMConcepts'][0]['Code'] if condition['ICD10CMConcepts'] else None,
            'Description': condition['ICD10CMConcepts'][0]['Description'] if condition['ICD10CMConcepts'] else None,
            'Score': condition['ICD10CMConcepts'][0]['Score'] if condition['ICD10CMConcepts'] else None
        })
    
    return pd.DataFrame(icd_data)

# Function to map medications to RxNorm codes using Comprehend Medical
def map_medications_to_rxnorm(text):
    comprehend_medical = boto3.client('comprehendmedical')
    rxnorm_response = comprehend_medical.infer_rx_norm(Text=text)
    rxnorm_entities = rxnorm_response['Entities']
    
    rxnorm_data = []
    for medication in rxnorm_entities:
        rxnorm_data.append({
            'Medication': medication.get('Text', ''),
            'RxNorm Code': medication['RxNormConcepts'][0]['Code'] if medication['RxNormConcepts'] else None,
            'Description': medication['RxNormConcepts'][0]['Description'] if medication['RxNormConcepts'] else None,
            'Score': medication['RxNormConcepts'][0]['Score'] if medication['RxNormConcepts'] else None
        })
    
    return pd.DataFrame(rxnorm_data)

# Function to detect PHI using Comprehend Medical
# def detect_phi(text):
    comprehend_medical = boto3.client('comprehendmedical')
    phi_response = comprehend_medical.detect_phi(Text=text)
    phi_entities = phi_response['Entities']
    
    phi_data = []
    for phi in phi_entities:
        phi_data.append({
            'Text': phi.get('Text', ''),
            'Category': phi.get('Category', ''),
            'Type': phi.get('Type', ''),
            'Score': phi.get('Score', '')
        })
    
    return pd.DataFrame(phi_data)

# Function to generate the prompt for decision task
def generate_prompt(icd_df, rxnorm_df, document_type):
    # Generate the part of the prompt based on the extracted data
    prompt = f"""
    Document Type: {document_type}

    Patient Medical Record Summary:

    Medical Conditions:
    {', '.join([f"{row['Condition']} (ICD-10: {row['ICD-10 Code']})" for _, row in icd_df.iterrows() if row['ICD-10 Code']])}

    Medications Prescribed:
    {', '.join([f"{row['Medication']} (RxNorm: {row['RxNorm Code']})" for _, row in rxnorm_df.iterrows() if row['RxNorm Code']])}

    """
    
    return prompt

def generate_non_medical_prompt(text, document_type):
    # Simple prompt generation based on the document type and extracted text
    if document_type == "PolicyDocument":
        prompt = f"""
        Document Type: {document_type}
        Summary:
        {text}
        """
    else:
        prompt = f"""
        Document Type: {document_type}
        Content Overview:
        {text}
        """
    return prompt

with st.sidebar:
   
    # Sidebar for folder creation or selection
    folder_name = st.sidebar.text_input("Enter folder name/Claim number")

    # Get existing folders from S3
    existing_folders = get_existing_folders(s3_bucket_name)

    # Dropdown for selecting an existing folder
    selected_folder = st.sidebar.selectbox("Or select an existing folder", options=["-- Select an existing folder --"] + existing_folders)

    # If a folder name is entered or selected, proceed to upload
    if folder_name.strip():
        folder_name = folder_name.strip()
        if not folder_name:
            st.warning("Please enter a valid folder name.")
    else:
        folder_name = selected_folder if selected_folder != "-- Select an existing folder --" else "new_folder"

    # File type selection
    file_type = st.selectbox("Select file type", [
        "APS",
        "Proof of Good Health",
        "Policy Document", 
        "Claim Form", 
        "Medical Diagnosis", 
        "Invoices", 
        "Proof of Loss", 
        "Supporting Documents", 
        "Photos/Images", 
        "Witness Statements", 
        "Correspondence", 
        "Other"
    ])

    # File upload
    uploaded_file = st.file_uploader("Upload a file", type=["png", "jpeg", "tiff", "pdf", "jpg"])

    # Upload file to the selected folder with metadata
    if uploaded_file is not None:
        if st.button("Upload File"):
            upload_file_to_s3(uploaded_file, folder_name, file_type)
    else:
        st.write("Please upload a file to proceed.")

# main Area
existing_folders = get_existing_folders(s3_bucket_name)

selected_folder = st.selectbox("Select existing folder",existing_folders, index=None)

# List files in the selected folder
if selected_folder:
    file_list = list_files_in_folder(s3_bucket_name, selected_folder)

    if file_list:

        for file in file_list:
          st.write( file)

        preview_required = st.checkbox("Document preview required",value=False)
        full_prompt=""

        # Requirement
        full_prompt += """
        
        Human:Based on the patient's medical conditions and prescribed medications, decide whether the claim should be approved or denied. Follow this format:

        1. Decision: [APPROVE/DENY]
        2. Reasoning: [Explain why based on the medical conditions and medications]
        3. Additional Documents Needed: [If applicable, specify additional documents required for decision-making]

        """
        # Form Type selection
        form_types = ["Plain Text Read", "Form Document Extraction"]
        selected_form_type = st.selectbox("Select the operation type", form_types, index=None)
        st.write("You selected the form type:", selected_form_type)

   
        if selected_form_type:
            for file_key in file_list:
                # Extract the document type from the suffix after the last underscore
                document_type = file_key.split('_')[-1].split('.')[0]  # Get the part after the last underscore

                # Display the file being processed
                file_name = os.path.basename(file_key)
                st.write(f"Processing file: {file_name}")

                if preview_required:                  
                    s3 = boto3.client('s3')
                    response = s3.get_object(Bucket=s3_bucket_name, Key=file_key)
                    file_stream = BytesIO(response['Body'].read())
                    st.image(file_stream)

                
                # Check if medical comprehension is needed
                if needs_medical_comprehension(document_type):
                    st.write(f"Processing medical document: {file_key}")

                    # Extract text from the file
                    text = extract_text_from_textract(s3_bucket_name, file_key, selected_form_type)
                    st.header(f"Data Extraction from {document_type} document...")
                    st.write(text)

                    # Perform medical entity extraction, ICD-10 mapping, RxNorm mapping, and PHI detection
                    medical_entities = extract_medical_entities(text)
                    icd_df = map_conditions_to_icd10(text)
                    rxnorm_df = map_medications_to_rxnorm(text)
                    # phi_df = detect_phi(text)

                    # Display extracted medical entities
                    st.subheader(f"Extracted Medical Entities from {document_type}")
                    st.dataframe(medical_entities)

                    # Display ICD-10 mappings
                    st.subheader(f"ICD-10-CM Code Mapping from {document_type}")
                    st.dataframe(icd_df)

                    # Display RxNorm mappings
                    st.subheader(f"RxNorm Code Mapping from {document_type}")
                    st.dataframe(rxnorm_df)

                    # Display PHI
                    # st.subheader(f"Protected Health Information (PHI) from {document_type}")
                    # st.dataframe(phi_df)

                    # Generate the prompt for this file and append it to the full prompt
                    prompt = generate_prompt(icd_df, rxnorm_df, document_type)
                    full_prompt += prompt + "\n\n"

                else:
                    st.write(f"Processing non-medical document: {file_key}")

                    # Only extract text from the non-medical file
                    text = extract_text_from_textract(s3_bucket_name, file_key, selected_form_type)
                    st.header(f"Data Extraction from {document_type} document...")
                    st.write(text)

                    # Generate a non-medical prompt (e.g., for policy documents)
                    non_medical_prompt = generate_non_medical_prompt(text, document_type)
                    full_prompt += non_medical_prompt + "\n\n"

               
         
            
            # Display the final prompt
            # st.header(f"Prompt")
            # st.text(full_prompt)

        
            st.header(f"Decision making")
            # Format the request payload using the model's native structure.
            request = {
                "inputText": full_prompt,  # Use the 'prompt' key as required
                "textGenerationConfig": {
                "maxTokenCount": 512,
                "temperature": 0.5,
              },  # Optional, adjust as needed
            }

           
            try:
                # Invoke the model with the request
                response = bedrock.invoke_model(modelId=model_id, body=json.dumps(request))
                # Decode the response body
                model_response = json.loads(response["body"].read())

                # Extract and print the response text
                response_text = model_response.get("results", "")
                
                for result in response_text:
                    output_text = result.get("outputText", "")
                    if output_text:
                        st.write(output_text)
                    else:
                        st.write("No outputText found in the result.")

            except (ClientError, Exception) as e:
                st.write(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
                exit(1)

    else:
        st.write(f"No files found in folder '{folder_name}'.")

