# Semiconductor Fabrication Workflow Generator

This repository helps you generate semiconductor fabrication workflows using Gemini CLI. Give it a final device image and some design notes, and it'll analyze the image, map colors to materials, and output a complete fabrication workflow with specific tools.

## What this does

- Uses Gemini CLI to analyze device images and generate fabrication flows
- Only uses tools from your `Tools_list.csv` 
- Maps image colors to materials using `color_mapping.py`
- The `wafer_training_data/` folder just has example images for reference

## Project Structure

```
Semi/
├── README.md
├── Tools_list.csv                 # Your available tools
├── color_mapping.py               # Color to material mapping
├── prompt_fabrication_workflow.txt# Gemini prompt
├── input_images/                  # Put device images here
└── wafer_training_data/           # Example images only
```

## Getting Started

1. Install and set up Gemini CLI
2. Put your device image(s) in `input_images/` 
3. Check that `Tools_list.csv` and `color_mapping.py` match your setup
4. Run Gemini with the prompt and files:

```bash
gemini run \
  --model gemini-1.5-pro \
  --prompt-file prompt_fabrication_workflow.txt \
  --file Tools_list.csv \
  --file color_mapping.py \
  --input-dir input_images \
  --output workflow.md
```

You'll get back a numbered fabrication workflow with tool selections, materials, and a visual diagram.

## What Gemini sees

- Device images from `input_images/`
- Tool list from `Tools_list.csv` 
- Color mapping from `color_mapping.py`
- Your text notes about materials and design

## Key files

- `Tools_list.csv`: Lists all available tools. Gemini only picks from here.
- `color_mapping.py`: Maps image colors to materials (Si, SiO2, Si3N4, Ti, Cu, Au, Al, empty)
- `wafer_training_data/`: Example images showing typical wafer structures

## The prompt

The full prompt is in `prompt_fabrication_workflow.txt`. It tells Gemini to:

- Analyze the device image
- Pick tools only from your list
- Generate numbered fabrication steps
- Include materials at each step
- Create a visual workflow diagram
- Skip exact process parameters (no times, temperatures, etc.)

## What you get back

- Numbered fabrication steps with process names and purposes
- Tool selections from your list only
- Materials used at each step
- Visual diagram showing the workflow

## Example images

The `wafer_training_data/` folder has example wafer images and metadata. These are just for reference to show what typical wafers look like.

---

**Status**: Ready to use with Gemini CLI  
**Updated**: August 2025