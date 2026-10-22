module load miniconda/23.11.0
conda activate /projectnb/multilm/jongin/anaconda3/envs/introspection

python plotters/plot_attention_srctoken_sweep.py outputs/runs/attn_endo_sweep_layer_src
