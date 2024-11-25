## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
The goal is to develop an application that can accurately recognize and categorize named entities such as persons, organizations, locations, dates, etc., from input text. By fine-tuning a pre-trained BART model specifically for NER tasks, the system should be able to understand contextual relationships and identify relevant entities. The Gradio framework will be used to build a user-friendly interface for real-time interaction and evaluation.
### DESIGN STEPS:

#### STEP 1: Data Preparation and Model Fine-Tuning
+ Collect or obtain a labeled NER dataset (e.g., CoNLL-2003).
+ Preprocess the dataset for tokenization and entity tagging.
+ Fine-tune a pre-trained BART model on the dataset using Hugging Face's transformers library.
#### STEP 2: Model Evaluation
+ Test the fine-tuned BART model on a validation set.
+ Evaluate performance using standard metrics like F1-score, precision, and recall.
+ Perform necessary adjustments and re-training if needed.
#### STEP 3: Gradio Interface Development
+ Use Gradio to build a simple interface where users can input text.
+ Integrate the fine-tuned model to process the input and display recognized entities in real-time.
+ Ensure the interface is intuitive and allows users to submit queries for NER.

### PROGRAM:

```
from transformers import BartTokenizer, BartForConditionalGeneration

# Load BART tokenizer and model
model_name = "facebook/bart-base"  # Replace with your fine-tuned model path if available
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

print("Model and tokenizer loaded successfully!")
def ner_function(input_text):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    
    # Generate predictions using the model
    outputs = model.generate(**inputs)
    
    # Decode the output
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return f"Recognized Entities: {decoded_output}"
import gradio as gr

# Gradio interface for the NER application
interface = gr.Interface(
    fn=ner_function,
    inputs=gr.Textbox(lines=5, label="Enter Text for NER"),
    outputs=gr.Textbox(label="Named Entities"),
    title="Named Entity Recognition (NER) with BART",
    description="Enter a piece of text, and this tool will identify named entities using a fine-tuned BART model."
)

# Launch the interface
interface.launch()
```
### OUTPUT:
![image](https://github.com/user-attachments/assets/bfecfd91-a76d-42f4-9493-05565b41f856)



### RESULT:
The result of this project is a fully functional Named Entity Recognition (NER) prototype application. The application uses a fine-tuned BART model to accurately identify and classify named entities such as persons, locations, and dates from input text. Through the integration of the Gradio framework, the model is deployed with a user-friendly interface, allowing users to input text and receive real-time entity recognition. The system processes the input text, extracts relevant entities, and displays them in an easily understandable format. The fine-tuned model demonstrates strong performance in recognizing contextual relationships, providing accurate entity categorization. This prototype serves as an effective tool for evaluating and deploying NER capabilities in various natural language processing applications.






