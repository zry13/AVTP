This repository is an anonymous code release for the paper “Adaptive Visual Token Pruning for Efficient Multi-Image Understanding in Large Visual Language Models”, provided for reproducibility verification during the review phase. After acceptance, we will update it to a complete version (including clearer documentation, cleaned scripts, etc.).

For the experiments of Qwen3VL and InternVL3.5, we use `transformers==4.57.1` and for the experiments of LLaVA-OneVison, we use the `transformers==4.43.1`.

You can use the `cat` command to overwrite the original source code with the modified model code, or you can import the modified model code directly in the evaluation script.

For example:

`
cat ./qwen3vl/modeling_qwen3vl_avtp.py > /usr/local/lib/python3.11/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py
`
