from model import GPT2Model
import argparse

parser = argparse.ArgumentParser(
        description = "Finetune OpenAI's GPT2 on Reddit Dataset"
)
parser.add_argument('--nsteps', help = 'Number of Steps for Training', default = 100)
parser.add_argument('--print_every', help = 'Number of Steps for Training', default = 10)
parser.add_argument('--sample_every', help = 'Number of Steps for Training', default = 200)
parser.add_argument('--save_every', help = 'Number of Steps for Training', default = 500)
parser.add_argument('--model_type',  help = "Which model to use for finetuning", default='124M')

args = parser.parse_args()
gpt2_model = GPT2Model(model_type = args.model_type)

# Start finetuning on first chunk
gpt2_model.fit(input_path = 'train_1.txt',
                   print_every = int(args.print_every),
                   sample_every = int(args.sample_every),
                   save_every = int(args.save_every),
                   num_steps = int(args.nsteps))

# Load the tuned model and finetune on remaining chunks
for i in range(2, 10):
    gpt2_model.fit(input_path = 'train_' + str(i) + '.txt',
                   overwrite = True,
                   restore_from = 'latest',
                   print_every = int(args.print_every),
                   sample_every = int(args.sample_every),
                   save_every = int(args.save_every),
                   num_steps = int(args.nsteps))
