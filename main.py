import argparse
import torch
import transformers

def transcribe(model):

    pass

def main ():

    parser = argparse.ArgumentParser("voxtral", description="A simple cli utility for using voxtral")
    parser.add_argument("filename")
    parser.add_argument("-M", "--model", default="mini", help="Which voxtral model to use.", choices=("mini", "base"))
    parser.add_argument("-m", "--mode", default="transcribe", choices=("transcribe", "translate", "custom"))
    parser.add_argument("-p", "--prompt", help="Custom prompt to use when using the custom mode")
    parser.add_argument("--repo_id", default="mistralai/Voxtral-Mini-3B-2507")
    parser.add_argument("--lang", default="en")

    args = parser.parse_args().__dict__

    # Verify args 
    if args['mode'] == 'custom' and args['prompt'] == None:
        print("Please provide a prompt")

        return

    repo_id = args['repo_id']
    device = "cuda"
    torch.set_default_device(device)
    
    processor = transformers.AutoProcessor.from_pretrained(repo_id)
    model = transformers.VoxtralForConditionalGeneration.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map=device)

    match args['mode']:
        case 'transcribe':
            inputs = processor.apply_transcrition_request(model_id=repo_id, language=[args['lang']], audio=args['filename'])
            inputs.to(torch.bfloat16)
            inputs.to(device)

            print(processor.batch_decode(model(**inputs)[:, inputs.input_ids.shape[1]:]))

            
if __name__ == "__main__":
    main()

