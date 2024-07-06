from pymongo import MongoClient, UpdateOne
from pymongo.server_api import ServerApi
from kaggle_secrets import UserSecretsClient
from datetime import datetime
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import threading
import math
import re


class RunInitializer:
    def __init__(self, mongo_url, db_name):
        self.client = MongoClient(mongo_url, server_api=ServerApi('1'))
        self.db = self.client[db_name]
    
    def initialize_run(self, source_collection_name, run_collection_name):
        source_collection = self.db[source_collection_name]
        run_collection = self.db[run_collection_name]
        
        documents = source_collection.find({}, {'_id': 0, 'question': 1, 'answer': 1})
        
        run_documents = []
        for document in documents:
            question = document['question']
            answer = document['answer']
            
            match = re.search(r'#### (-?\d+)', answer)
            correct_answer = match.group(1) if match else ""
            
            run_documents.append({
                "question": question,
                "response": "",
                "instance": "",
                "status": "pending",
                "correct_answer": correct_answer,
                "generated_answer": ""
            })
        
        try:
            run_collection.insert_many(run_documents)
            print("Run table created and data inserted successfully!")
        except Exception as e:
            print(f"An error occurred while creating the run table: {e}")
    

        
class RunInitializerRetry:
    def __init__(self, mongo_url, db_name):
        self.client = MongoClient(mongo_url, server_api=ServerApi('1'))
        self.db = self.client[db_name]
    
    def initialize_retry_run(self, source_collection_name, retry_collection_name):
        source_collection = self.db[source_collection_name]
        retry_collection = self.db[retry_collection_name]
        
        default_fields = {
            'response_2': '',
            'generated_answer_2': '',
            'status': 'pending',
            'instance': ''
        }
        
        incorrect_docs = []
        
        for doc in source_collection.find():
            try:
                doc.pop('start_time', None)
                
                if 'correct_answer' in doc and 'generated_answer' in doc:
                    if float(doc['correct_answer']) != float(doc['generated_answer']):
                        doc.update(default_fields)
                        incorrect_docs.append(doc)
                else:
                    doc.update(default_fields)
                    incorrect_docs.append(doc)
            except (ValueError, KeyError):
                doc.update(default_fields)
                incorrect_docs.append(doc)
        
        if incorrect_docs:
            try:
                retry_collection.insert_many(incorrect_docs)
                print(f"Inserted {len(incorrect_docs)} documents into {retry_collection_name} collection.")
            except Exception as e:
                print(f"An error occurred while inserting documents into {retry_collection_name} collection: {e}")


class DatabaseHandler:
    def __init__(self, mongo_url, db_name, run_collection_name):
        self.client = MongoClient(mongo_url, server_api=ServerApi('1'))
        self.db = self.client[db_name]
        self.collection = self.db[run_collection_name]
        
    def fetch_batch(self, batch_size, instance_name):
        current_time = datetime.utcnow()
        pipeline = [
            {"$match": {"status": "pending"}},
            {"$limit": batch_size},
            {"$set": {"status": "in progress", "instance": instance_name, "start_time": current_time}},
            {"$merge": {"into": self.collection.name, "whenMatched": "merge", "whenNotMatched": "fail"}}
        ]
        self.collection.aggregate(pipeline)
        batch = self.collection.find({"status": "in progress", "instance": instance_name})
        return list(batch)
        
    def update_responses(self, responses, instance_name):
        bulk_ops = []
        for doc_id, response, generated_answer in responses:
            bulk_ops.append(UpdateOne(
                {"_id": doc_id},
                {
                    "$set": {
                        "response": response,
                        "generated_answer": generated_answer,
                        "status": "completed",
                        "instance": instance_name
                    }
                }
            ))
        self.collection.bulk_write(bulk_ops)
        print(f"Updated responses for instance {instance_name}")
        
class DatabaseHandlerRetry:
    def __init__(self, mongo_url, db_name, run_collection_name):
        self.client = MongoClient(mongo_url, server_api=ServerApi('1'))
        self.db = self.client[db_name]
        self.collection = self.db[run_collection_name]
        
    def fetch_batch(self, batch_size, instance_name):
        current_time = datetime.utcnow()
        pipeline = [
            {"$match": {"status": "pending"}},
            {"$limit": batch_size},
            {"$set": {"status": "in progress", "instance": instance_name, "start_time": current_time}},
            {"$merge": {"into": self.collection.name, "whenMatched": "merge", "whenNotMatched": "fail"}}
        ]
        self.collection.aggregate(pipeline)
        batch = self.collection.find({"status": "in progress", "instance": instance_name})
        return list(batch)
        
    def update_responses(self, responses, instance_name):
        bulk_ops = []
        for doc_id, response, generated_answer in responses:
            bulk_ops.append(UpdateOne(
                {"_id": doc_id},
                {
                    "$set": {
                        "response_2": response,
                        "generated_answer_2": generated_answer,
                        "status": "completed",
                        "instance": instance_name
                    }
                }
            ))
        self.collection.bulk_write(bulk_ops)
        print(f"Updated responses for instance {instance_name}")
        
        
        
class ModelHandler:
    def __init__(self, model_path, device):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            device_map=device,
            quantization_config=quantization_config
        )
        self.device = device

    def tokenize_inputs(self, data):
        questions = [item['tagged_question'] for item in data]
        return self.tokenizer(questions, return_tensors="pt", padding=True, truncation=True).to(self.device)
    
    def generate_responses(self, inputs, max_new_tokens, temperature, top_p, repetition_penalty, num_return_sequences):
        responses = []
        batch_input_ids = inputs.input_ids.to(self.device)
        batch_attention_mask = inputs.attention_mask.to(self.device)
        batch_responses = self.model.generate(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences
        )
        decoded_responses = [self.tokenizer.decode(response, skip_special_tokens=True) for response in batch_responses]
        responses.extend(decoded_responses)
        torch.cuda.empty_cache()
        return responses
        
    def generate_answers(self, inputs, batch_size, max_new_tokens, temperature, top_p, repetition_penalty, num_return_sequences):
        responses = []
        num_batches = math.ceil(len(inputs.input_ids) / batch_size)
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(inputs.input_ids))
            batch_input_ids = inputs.input_ids[start_idx:end_idx].to(self.device)
            batch_attention_mask = inputs.attention_mask[start_idx:end_idx].to(self.device)
            batch_responses = self.model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences
            )
            decoded_responses = [self.tokenizer.decode(response, skip_special_tokens=True) for response in batch_responses]
            responses.extend(decoded_responses)
            torch.cuda.empty_cache()
        return responses
    
    
class ModelHandlerPeft:
    def __init__(self, model_path, peft_path, device):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            device_map=device,
            quantization_config=quantization_config
        )
        self.model = PeftModel.from_pretrained(self.base_model,peft_path)
        self.device = device

    def tokenize_inputs(self, data):
        questions = [item['tagged_question'] for item in data]
        return self.tokenizer(questions, return_tensors="pt", padding=True, truncation=True).to(self.device)
    
    def generate_responses(self, inputs, max_new_tokens, temperature, top_p, repetition_penalty, num_return_sequences):
        responses = []
        batch_input_ids = inputs.input_ids.to(self.device)
        batch_attention_mask = inputs.attention_mask.to(self.device)
        batch_responses = self.model.generate(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences
        )
        decoded_responses = [self.tokenizer.decode(response, skip_special_tokens=True) for response in batch_responses]
        responses.extend(decoded_responses)
        torch.cuda.empty_cache()
        return responses
        
    def generate_answers(self, inputs, batch_size, max_new_tokens, temperature, top_p, repetition_penalty, num_return_sequences):
        responses = []
        num_batches = math.ceil(len(inputs.input_ids) / batch_size)
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(inputs.input_ids))
            batch_input_ids = inputs.input_ids[start_idx:end_idx].to(self.device)
            batch_attention_mask = inputs.attention_mask[start_idx:end_idx].to(self.device)
            batch_responses = self.model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences
            )
            decoded_responses = [self.tokenizer.decode(response, skip_special_tokens=True) for response in batch_responses]
            responses.extend(decoded_responses)
            torch.cuda.empty_cache()
        return responses
    
    
class ResponseProcessor:
    def __init__(self, one_shot_prompt):
        self.one_shot_prompt = one_shot_prompt
        
    def transform_response(self, response):
        try:
            question_part = response.split('[INST]')[1].split('[/INST]')[0].strip()
            answer_part = response.split('[/INST]')[1].strip()
            return f"#question: {question_part}\n#answer: {answer_part}"
        except Exception as e:
            print(f"Error processing response: {e}")
            return response

    def transform_responses(self, responses):
        return [self.transform_response(response) for response in responses]

    def integrate_one_shot_prompt(self, transformed_responses):
        prompts = []
        for text in transformed_responses:
            prompt = self.one_shot_prompt.format(text=text)
            prompts.append(prompt)
        return prompts

    def extract_generated_answer(self, text):
        try:
            split_text = text.rsplit("#final answer as number:", 1)
            if len(split_text) > 1:
                return split_text[1].strip().split()[0]
            else:
                return ""
        except Exception as e:
            print(f"Error extracting final answer: {e}")
            return ""
        
class ResponseProcessorRetry:
    def __init__(self, one_shot_prompt,retry_prompt):
        self.one_shot_prompt = one_shot_prompt
        self.retry_prompt = retry_prompt
        
    def transform_response(self, response):
        try:
            question_part = response.split('[INST]')[1].split('[/INST]')[0].strip()
            answer_part = response.split('[/INST]')[2].strip()
            return f"#question: {question_part}\n#answer: {answer_part}"
        except Exception as e:
            print(f"Error processing response: {e}")
            return response

    def transform_responses(self, responses):
        return [self.transform_response(response) for response in responses]

    def integrate_one_shot_prompt(self, transformed_responses):
        prompts = []
        for text in transformed_responses:
            prompt = self.one_shot_prompt.format(text=text)
            prompts.append(prompt)
        return prompts

    def extract_generated_answer(self, text):
        try:
            split_text = text.rsplit("#final answer as number:", 1)
            if len(split_text) > 1:
                return split_text[1].strip().split()[0]
            else:
                return ""
        except Exception as e:
            print(f"Error extracting final answer: {e}")
            return ""
        
class RunManager:
    def __init__(self, db_handler, model_handler_0, model_handler_1, response_processor, instance_name_0, instance_name_1, batch_size):
        self.db_handler = db_handler
        self.model_handler_0 = model_handler_0
        self.model_handler_1 = model_handler_1
        self.response_processor = response_processor
        self.instance_name_0 = instance_name_0
        self.instance_name_1 = instance_name_1
        self.batch_size = batch_size
        
    def has_pending_batches(self):
        pending_count = self.db_handler.collection.count_documents({"status": "pending"})
        return pending_count > 0
    
    def process_batch(self, model_handler, instance_name):
        while self.has_pending_batches():
            batch = self.db_handler.fetch_batch(self.batch_size, instance_name)
            data = [{"_id": doc["_id"], "tagged_question": f"<s>[INST]{doc['question']} \n do it step by step[/INST] "} for doc in batch]
            inputs = model_handler.tokenize_inputs(data)
            responses = model_handler.generate_responses(inputs, 1024, 0.01, 0.1, 1.2, 1)
            transformed_responses = self.response_processor.transform_responses(responses)
            prompts = self.response_processor.integrate_one_shot_prompt(transformed_responses)
            extract_inputs = model_handler.tokenizer(prompts,return_tensors="pt", padding=True, truncation=True).to(model_handler.device)
            generated_answers = model_handler.generate_answers(extract_inputs, (int(self.batch_size/2)), 10, 0.01, 0.1, 1, 1)
            bulk = [(data[i]["_id"], responses[i], self.response_processor.extract_generated_answer(generated_answers[i])) for i in range(len(responses))]
            self.db_handler.update_responses(bulk, instance_name)

    def main_loop(self):
        threads = []
        t0 = threading.Thread(target=self.process_batch, args=(self.model_handler_0, self.instance_name_0))
        t1 = threading.Thread(target=self.process_batch, args=(self.model_handler_1, self.instance_name_1))
        threads.append(t0)
        threads.append(t1)
        t0.start()
        t1.start()

        for t in threads:
            t.join()
            
            
class RunManagerRetry:
    def __init__(self, db_handler, model_handler_0, model_handler_1, response_processor, instance_name_0, instance_name_1, batch_size):
        self.db_handler = db_handler
        self.model_handler_0 = model_handler_0
        self.model_handler_1 = model_handler_1
        self.response_processor = response_processor
        self.instance_name_0 = instance_name_0
        self.instance_name_1 = instance_name_1
        self.batch_size = batch_size
        
    def has_pending_batches(self):
        pending_count = self.db_handler.collection.count_documents({"status": "pending"})
        return pending_count > 0
    
    def process_batch(self, model_handler, instance_name):
        while self.has_pending_batches():
            batch = self.db_handler.fetch_batch(self.batch_size, instance_name)
            data = [{"_id": doc["_id"], "tagged_question": f"<s>{doc['response']} {self.response_processor.retry_prompt} "} for doc in batch]
            inputs = model_handler.tokenize_inputs(data)
            responses = model_handler.generate_responses(inputs, 1024, 0.01, 0.1, 1.2, 1)
            transformed_responses = self.response_processor.transform_responses(responses)
            prompts = self.response_processor.integrate_one_shot_prompt(transformed_responses)
            extract_inputs = model_handler.tokenizer(prompts,return_tensors="pt", padding=True, truncation=True).to(model_handler.device)
            generated_answers = model_handler.generate_answers(extract_inputs, (int(self.batch_size/2)), 10, 0.01, 0.1, 1, 1)
            bulk = [(data[i]["_id"], responses[i], self.response_processor.extract_generated_answer(generated_answers[i])) for i in range(len(responses))]
            self.db_handler.update_responses(bulk, instance_name)

    def main_loop(self):
        threads = []
        t0 = threading.Thread(target=self.process_batch, args=(self.model_handler_0, self.instance_name_0))
        t1 = threading.Thread(target=self.process_batch, args=(self.model_handler_1, self.instance_name_1))
        threads.append(t0)
        threads.append(t1)
        t0.start()
        t1.start()

        for t in threads:
            t.join()
