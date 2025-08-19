# GEMINI.md - Semiconductor Fabrication Workflow Customization

## Project Overview
This is a semiconductor fabrication workflow project that focuses on wafer processing, material analysis, and fabrication process visualization. The project contains tools for generating and analyzing semiconductor wafer images, scaling analysis, and workflow automation.

## Key Project Components

### Core Files
- `README.md` - Project documentation and setup instructions
- `prompt_fabrication_workflow.txt` - Fabrication workflow prompts and templates
- `Tools_list.csv` - List of available fabrication tools and processes
- `color_mapping.py` - Color mapping utilities for wafer visualization
- `requirements.txt` - Python dependencies

### Main Directories
- `wafer_training_data/` - Contains generated wafer images and metadata (1000+ images)
- `scaling/` - Scaling analysis tools and results
- `input_images/` - Input images for processing

## Customization Guidelines for Gemini

### 1. Semiconductor Process Focus
- Always assume Si (silicon) wafer as the default starting material unless specified otherwise
- Focus on standard semiconductor fabrication processes: deposition, etching, lithography, doping, etc.
- Use industry-standard terminology and process flows

### 2. Visualization Preferences
- Flow charts should always start with a silicon (Si) substrate
- Use arrows to connect process steps (deposition → etch → etc.)
- 2D wafer representations should reflect each step continuously
- Maintain visual consistency across process flows

### 3. Process Documentation Style
- **DO NOT include**: temperature, duration, equipment details, or lengthy descriptions
- **DO include**: material names, process types, layer sequences
- Keep process steps concise and focused on the essential workflow

### 4. Code Generation Guidelines
- Use Python for wafer image generation and analysis
- Implement proper color mapping for different materials
- Include error handling for fabrication process simulations
- Follow semiconductor industry coding standards

### 5. Data Handling
- The wafer_training_data directory contains 1000+ generated wafer images
- Focus on process workflow rather than individual image analysis
- Use metadata files for process information rather than analyzing all images
- Generate new wafer images based on process parameters rather than training on existing ones

### 6. Tool Integration
- Reference Tools_list.csv for available fabrication processes
- Use color_mapping.py for consistent material visualization
- Integrate with scaling analysis tools when relevant
- Maintain compatibility with existing workflow automation

### 7. Response Format
- Provide clear, step-by-step process flows
- Include visual representations when helpful
- Use semiconductor industry terminology
- Keep responses focused on fabrication workflow rather than detailed analysis

## Example Interaction Style
When asked about a fabrication process:
1. Start with Si substrate
2. List process steps with arrows: Si → SiO2 deposition → Photoresist coating → Lithography → Etch → Strip
3. Show 2D wafer cross-section at each step
4. Focus on materials and process types, not parameters

## File Structure Awareness
- Training data is extensive (1000+ images) - focus on process workflows
- Use existing tools and utilities rather than recreating functionality
- Reference project files for context and consistency
- Maintain project structure and organization

This customization ensures Gemini provides relevant, focused assistance for semiconductor fabrication workflows while respecting the project's scope and avoiding overwhelming data analysis.
