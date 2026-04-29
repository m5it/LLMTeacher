#!/bin/bash
cd /home/t3ch/models/llmteacher

# Set learning rates: lr=1e-4, min_lr=5e-4 (Option C)
python3 -c "
import json
config = json.loads(open('config/training_config.json').read())
config['learning_rate'] = 1e-4
config['min_lr'] = 5e-4
open('config/training_config.json', 'w').write(json.dumps(config, indent=2))
print('Set learning_rate=1e-4, min_lr=5e-4')
print(f'Current config: {json.dumps(config, indent=2)}')
"

# Start test training with 50k iterations
echo "Starting test training (50k iterations)..."
nohup python3 llmteacher.py train --output data/models/test_run_50k.pt > training_test.log 2>&1 &
echo "Training started in background. Monitor with: tail -f training_test.log"
