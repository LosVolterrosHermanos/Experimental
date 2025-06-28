This code is for generating the gpt2(124M*)-hummingbird.

(This GPT2 is not using weight tying, hence the actual parameter count is 160M).

You should have:
grab_fineweb.py
nanogpt_log_losses_dana.py
nanogpt_minimal.py
optimizers.py
plot_sweep_results.py
Sweep.sh

Setup:
1) You'll need a python environment with jax, flax, optax, tiktoken, tqdm, panda, datasets. (I may be missing a few)
2) Run grab_fineweb.py, which will download the fineweb-edu text.
3) Try running the training loop:
nanogpt_log_losses_dana.py --train_steps=100 --batch_size=32 --val_batch_size=32 --seq_len=32 --dana_g2=0.05 --dana_g3_iv=0.01 --dana_g3_p=-$p
(seq_len=1024 is the default)
4) Sweep.sh is setup to generate the hummingbird loss curve data.  Change the train_steps=10000 for the short ~10x(30 minute) version.  I think we need train_steps=100000 for a full picture.
5) plot_sweep_results.py will generate the figure.  Look on line 19 to change which set of loss curves it uses in the plot.

