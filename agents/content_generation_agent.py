import logging
import json
import re
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from .base_agent import BaseAgent
import base64
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from docx import Document
from docx.shared import Inches as DocxInches
import io
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class ContentGenerationAgent(BaseAgent):
    """Agent specialized in flexible content generation with multiple output capabilities"""
    
    CONTENT_TYPES = {
        "presentation": ["powerpoint", "presentacion", "slides", "pptx", "diapositivas"],
        "document": ["documento", "write", "escribe", "doc", "docx", "texto", "word"],
        "visualization": ["chart", "graph", "grafico", "plot", "visualizacion", "figura"],
        "infographic": ["infografia", "infographic", "visual", "imagen"],
        "report": ["reporte", "report", "informe", "análisis"],
        "summary": ["resumen", "summary", "síntesis"],
        "code": ["codigo", "code", "script", "programa"],
        "email": ["correo", "email", "mail"],
        "social": ["post", "publicacion", "tweet", "linkedin"]
    }

    def __init__(self, task: str, openai_api_key: str, metadata: Optional[Dict[str, Any]] = None, partial_data: Optional[Dict[str, Any]] = None):
        super().__init__(task, openai_api_key, metadata)
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4-turbo",
            temperature=0.7
        )
        self.partial_data = partial_data or {}
        self.content_type = self._determine_content_type()
        self.format_handlers = {
            "presentation": self._create_presentation,
            "document": self._create_document,
            "visualization": self._create_visualization,
            "infographic": self._create_infographic,
            "report": self._create_report,
            "summary": self._create_summary,
            "code": self._create_code,
            "email": self._create_email,
            "social": self._create_social_content
        }

    def _determine_content_type(self) -> str:
        """Determine content type from task description"""
        task_lower = self.task.lower()
        
        for content_type, keywords in self.CONTENT_TYPES.items():
            if any(keyword in task_lower for keyword in keywords):
                return content_type
                
        return self._infer_content_type(task_lower)

    def _infer_content_type(self, task: str) -> str:
        """Infer content type based on task context and patterns"""
        # Length-based inference
        if len(task.split()) <= 10 and any(word in task for word in ["tweet", "post"]):
            return "social"
            
        # Analysis-based inference    
        if any(word in task for word in ["analiza", "analyze", "compara", "compare"]):
            return "report"
            
        # Visual-based inference
        if any(word in task for word in ["muestra", "show", "visualiza", "display"]):
            return "visualization"
            
        return "document"

    async def _execute(self) -> Dict[str, Any]:
        """Execute content generation with enhanced flexibility"""
        try:
            # Gather input data from context and other agents
            content_data = await self._gather_content_data()
            if not content_data:
                content_data = await self._gather_content_data_fallback()
            
            # Create content plan based on type and data
            content_plan = await self._create_content_plan(content_data)
            
            # Generate base content
            content = await self._generate_content(content_plan)
            
            # Format according to type using appropriate handler
            if self.content_type in self.format_handlers:
                handler = self.format_handlers[self.content_type]
                formatted_content = await handler(content)
            else:
                formatted_content = {"content": content, "format": "text"}
            
            return {
                "result": formatted_content,
                "content_type": self.content_type,
                "plan": content_plan,
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "content_structure": content.get("structure"),
                    "source_data": content_data.get("sources", [])
                }
            }

        except Exception as e:
            logger.error(f"Error in content generation: {e}")
            return {"error": str(e)}

    async def _gather_content_data(self) -> Dict[str, Any]:
        """Gather necessary data for content generation"""
        try:
            research_needed = [
                "What key information is needed",
                "What are the main topics to cover",
                "What examples or use cases to include",
                "What is the target audience",
                "What style and tone to use"
            ]
            
            research_results = await self._conduct_research(research_needed)
            return research_results
            
        except Exception as e:
            logger.error(f"Error gathering content data: {e}")
            return None

    async def _gather_content_data_fallback(self) -> Dict[str, Any]:
        """Fallback method for gathering content data dynamically"""
        try:
            messages = [
                {"role": "system", "content": """Analyze the given task and generate a basic content structure.
                Include key sections, main points, and relevant aspects that should be covered.
                Be specific to the topic but maintain a logical structure."""},
                {"role": "user", "content": self.task}
            ]
            
            response = await self.llm.agenerate([messages])
            content_structure = self._parse_llm_response(response.generations[0][0].message.content)
            
            return {
                "key_points": content_structure.get("key_points", []),
                "content": content_structure.get("content", {}),
                "source": "dynamic_generation"
            }
        except Exception as e:
            logger.error(f"Error in fallback content generation: {e}")
            return self._create_minimal_content_structure()

    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM response into structured content"""
        try:
            # First try to parse as JSON
            if content.strip().startswith('{'):
                return json.loads(content)
                
            # Otherwise parse markdown-style content
            structure = {
                "title": "",
                "sections": []
            }
            
            current_section = None
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('# '):
                    structure["title"] = line.lstrip('# ')
                elif line.startswith('## '):
                    if current_section:
                        structure["sections"].append(current_section)
                    current_section = {
                        "title": line.lstrip('## '),
                        "content": [],
                        "bullets": []
                    }
                elif line.startswith('- ') and current_section:
                    current_section["bullets"].append(line.lstrip('- '))
                elif current_section:
                    current_section["content"].append(line)
                    
            if current_section:
                structure["sections"].append(current_section)
                
            return {
                "key_points": [s["title"] for s in structure["sections"]],
                "content": structure
            }
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return self._create_minimal_content_structure()

    def _create_minimal_content_structure(self) -> Dict[str, Any]:
        """Create a minimal content structure based on task analysis"""
        task_terms = self._extract_key_terms(self.task)
        
        structure = {
            "title": self.task,
            "sections": [
                {
                    "title": "Introducción",
                    "content": [f"Análisis de: {self.task}"],
                    "bullets": []
                },
                {
                    "title": "Desarrollo",
                    "content": [f"Aspectos principales sobre {task_terms[0] if task_terms else 'el tema'}"],
                    "bullets": []
                },
                {
                    "title": "Conclusión",
                    "content": ["Puntos clave y consideraciones finales"],
                    "bullets": []
                }
            ]
        }
        
        return {
            "key_points": [s["title"] for s in structure["sections"]],
            "content": structure
        }

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for structuring content"""
        common_words = {"que", "es", "la", "el", "en", "de", "para", "por", "con", "un", "una", "sobre"}
        words = text.lower().split()
        return [word for word in words if word not in common_words and len(word) > 2]

    async def _create_content_plan(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a dynamic content plan based on task and type"""
        messages = [
            {"role": "system", "content": f"""Create a detailed plan for a {self.content_type} about the given topic.
            Consider:
            1. Key information to present
            2. Effective structure
            3. Appropriate style and tone
            4. Enhancing visual elements
            5. Key messages to emphasize
            
            The plan should be comprehensive but flexible."""},
            {"role": "user", "content": f"Task: {self.task}\nContent Data: {json.dumps(content_data)}"}
        ]
        
        try:
            response = await self.llm.agenerate([messages])
            return self._parse_plan(response.generations[0][0].message.content)
        except Exception as e:
            logger.error(f"Error creating content plan: {e}")
            return self._create_basic_plan()

    def _create_basic_plan(self) -> Dict[str, Any]:
        """Create a basic content plan when detailed planning fails"""
        return {
            "overview": [f"Content about: {self.task}"],
            "structure": [
                "Introduction to topic",
                "Main concepts development",
                "Examples and applications",
                "Conclusions and insights"
            ],
            "style": [
                "Professional",
                "Clear and concise",
                "Audience focused"
            ],
            "visual_elements": [
                "Explanatory diagrams",
                "Highlighted key points",
                "Relevant visuals"
            ]
        }

    def _parse_plan(self, content: str) -> Dict[str, Any]:
        """Parse the content plan"""
        try:
            if content.startswith('{'):
                return json.loads(content)
        except:
            pass

        # Fallback parsing
        sections = ["overview", "structure", "style", "visual_elements", "key_messages"]
        plan = {section: [] for section in sections}
        current_section = None

        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue

            lower_line = line.lower()
            if any(section in lower_line for section in sections):
                current_section = next(s for s in sections if s in lower_line)
            elif current_section and line.startswith('-'):
                plan[current_section].append(line.lstrip('- '))

        return plan

    async def _generate_content(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content based on plan"""
        messages = [
            {"role": "system", "content": f"""Create detailed {self.content_type} content for: {self.task}
    Include:
    - Clear title and structure
    - Comprehensive sections
    - Key points and explanations
    - Proper formatting
    Format as JSON with sections array containing title, content, and bullets."""},
            {"role": "user", "content": self.task}
        ]

        response = await self.llm.agenerate([messages])
        content = response.generations[0][0].message.content

        # Parse and structure the response
        try:
            if content.strip().startswith('{'):
                return json.loads(content)
            return self._structure_content(content)
        except:
            return self._structure_content(content)

    def _create_dynamic_prompt(self, plan: Dict[str, Any]) -> str:
        """Create a dynamic prompt based on content type and plan"""
        base_prompt = f"""Generate comprehensive content for a {self.content_type} about the given topic.

Content Plan:
{json.dumps(plan, indent=2)}

Requirements based on content type:"""

        type_specific_requirements = {
            "presentation": """
- Create clear, concise slides
- Include engaging visual descriptions
- Break down complex ideas
- Use bullet points effectively
- Maintain presentation flow""",
            
            "document": """
- Develop detailed paragraphs
- Include proper citations
- Maintain formal structure
- Use clear headings
- Provide thorough explanations""",
            
            "visualization": """
- Describe data relationships
- Specify chart types needed
- Include axis labels and titles
- Define color schemes
- Note key data points""",
            
            "infographic": """
- Create visual hierarchy
- Include key statistics
- Design flow of information
- Specify icon needs
- Balance text and visuals""",
            
            "report": """
- Include executive summary
- Provide detailed analysis
- Include methodology
- Present findings clearly
- Add recommendations""",
            
            "summary": """
- Focus on key points
- Maintain brevity
- Capture essential ideas
- Use clear structure
- Highlight conclusions""",
            
            "code": """
- Specify language requirements
- Include comments
- Structure functions logically
- Handle edge cases
- Include usage examples""",
            
            "email": """
- Create clear subject line
- Maintain professional tone
- Structure content logically
- Include call to action
- Close appropriately""",
            
            "social": """
- Create engaging hooks
- Use appropriate tone
- Include hashtags
- Optimize length
- Add call to action"""
        }

        return base_prompt + type_specific_requirements.get(self.content_type, """
- Maintain clear structure
- Present information logically
- Include relevant details
- Ensure completeness
- Focus on clarity""")

    async def _create_presentation(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Create a PowerPoint presentation"""
        try:
            prs = Presentation()
            
            # Title slide
            title_slide = prs.slides.add_slide(prs.slide_layouts[0])
            title = content.get("title", "Inteligencia Artificial")
            subtitle = content.get("subtitle", "Una Introducción Comprensiva")
            title_slide.shapes.title.text = title
            if hasattr(title_slide, "placeholders") and len(title_slide.placeholders) > 1:
                title_slide.placeholders[1].text = subtitle

            # Content slides
            for section in content.get("sections", []):
                slide = prs.slides.add_slide(prs.slide_layouts[1])
                if hasattr(slide.shapes, "title"):
                    slide.shapes.title.text = section.get("title", "")
                
                # Add content
                if hasattr(slide.shapes, "placeholders") and len(slide.shapes.placeholders) > 1:
                    body_shape = slide.shapes.placeholders[1]
                    tf = body_shape.text_frame
                    
                    # Add main content
                    if section.get("content"):
                        p = tf.add_paragraph()
                        p.text = "\n".join(section.get("content", []))
                    
                    # Add bullets
                    for bullet in section.get("bullets", []):
                        p = tf.add_paragraph()
                        p.text = bullet
                        p.level = 1

            # Save to bytes
            pptx_buffer = io.BytesIO()
            prs.save(pptx_buffer)
            pptx_buffer.seek(0)
            
            return {
                "content": base64.b64encode(pptx_buffer.getvalue()).decode(),
                "format": "pptx",
                "title": title
            }
            
        except Exception as e:
            logger.error(f"Error creating presentation: {e}")
            return {"error": str(e)}

    async def _create_document(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Create a Word document"""
        try:
            doc = Document()
            
            # Add title
            doc.add_heading(content["title"], 0)
            if "subtitle" in content:
                doc.add_heading(content["subtitle"], 1)
                
            # Add sections
            for section in content.get("sections", []):
                doc.add_heading(section["title"], 2)
                
                # Add content
                if section.get("content"):
                    doc.add_paragraph("\n".join(section["content"]))
                
                # Add bullets
                if section.get("bullets"):
                    for bullet in section["bullets"]:
                        doc.add_paragraph(bullet, style='List Bullet')

            # Save to bytes and convert to base64
            docx_buffer = io.BytesIO()
            doc.save(docx_buffer)
            docx_buffer.seek(0)
            
            return {
                "content": base64.b64encode(docx_buffer.getvalue()).decode(),
                "format": "docx",
                "title": content.get("title", "Document")
            }
            
        except Exception as e:
            logger.error(f"Error creating document: {e}")
            return {"error": str(e)}
        
    async def _create_code(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Create code content based on the structure"""
        try:
            code_parts = []
            
            # Add header comment with title
            title = content.get("title", "Generated Code")
            code_parts.append(f'"""\n{title}\n{"-" * len(title)}\n"""')
            
            # Add imports if in first section
            if content.get("sections"):
                first_section = content["sections"][0]
                if "import" in first_section.get("title", "").lower():
                    code_parts.extend(first_section.get("content", []))
                    code_parts.append("")  # Add spacing after imports
            
            # Process each section
            for section in content.get("sections", []):
                # Skip import section as it's already handled
                if "import" in section.get("title", "").lower():
                    continue
                    
                # Add section header as comment
                if section.get("title"):
                    code_parts.append(f"\n# {section['title']}")
                    
                # Add section content
                if section.get("content"):
                    code_parts.extend(section["content"])
                    
                # Add bulleted items as individual lines
                if section.get("bullets"):
                    code_parts.extend(section["bullets"])
                    
                # Add spacing between sections
                code_parts.append("")
            
            # Join all parts with proper line breaks
            final_code = "\n".join(code_parts)
            
            # Try to determine the language
            language = "python"  # Default to Python
            if "javascript" in content.get("title", "").lower() or "js" in content.get("title", "").lower():
                language = "javascript"
            elif "java" in content.get("title", "").lower():
                language = "java"
            elif "cpp" in content.get("title", "").lower() or "c++" in content.get("title", "").lower():
                language = "cpp"
            
            return {
                "content": final_code,
                "format": "code",
                "language": language,
                "title": content.get("title", "Generated Code")
            }
                
        except Exception as e:
            logger.error(f"Error creating code: {e}")
            return {"error": str(e)}
    
    async def _conduct_research(self, questions: List[str]) -> Dict[str, Any]:
        """Conduct research to gather information for content generation"""
        try:
            # Create research prompts
            messages = [
                {"role": "system", "content": f"""Research and provide detailed answers about {self.task}.
                Consider:
                - Current and accurate information
                - Key concepts and definitions
                - Important examples and applications
                - Relevant context and background
                
                Structure your response to address each question thoroughly but concisely."""},
                {"role": "user", "content": "\n".join(f"- {q}" for q in questions)}
            ]

            response = await self.llm.agenerate([messages])
            research_content = response.generations[0][0].message.content

            # Parse research results
            results = {
                "key_points": [],
                "examples": [],
                "definitions": {},
                "context": [],
                "sources": []
            }

            current_section = None
            for line in research_content.split('\n'):
                line = line.strip()
                if not line:
                    continue

                # Try to identify sections
                lower_line = line.lower()
                if "key point" in lower_line or "main point" in lower_line:
                    current_section = "key_points"
                    continue
                elif "example" in lower_line or "case study" in lower_line:
                    current_section = "examples"
                    continue
                elif "definition" in lower_line or "means" in lower_line:
                    current_section = "definitions"
                    continue
                elif "context" in lower_line or "background" in lower_line:
                    current_section = "context"
                    continue
                elif "source" in lower_line or "reference" in lower_line:
                    current_section = "sources"
                    continue

                # Add content to appropriate section
                if line.startswith('- ') or line.startswith('* '):
                    line = line.lstrip('- ').lstrip('* ').strip()
                    if current_section == "definitions":
                        if ':' in line:
                            term, definition = line.split(':', 1)
                            results["definitions"][term.strip()] = definition.strip()
                    elif current_section and current_section in results:
                        results[current_section].append(line)
                elif current_section and current_section in results and line:
                    if isinstance(results[current_section], list):
                        results[current_section].append(line)

            # Add any available data from partial_data
            if self.partial_data:
                for key, value in self.partial_data.items():
                    if isinstance(value, dict) and "content" in value:
                        # Add relevant content from other agents
                        if "key_points" in value:
                            results["key_points"].extend(value["key_points"])
                        if "examples" in value:
                            results["examples"].extend(value["examples"])
                        if "context" in value:
                            results["context"].extend(value["context"])

            return results

        except Exception as e:
            logger.error(f"Error conducting research: {e}")
            return {
                "key_points": [f"Main aspects of {self.task}"],
                "examples": [],
                "definitions": {},
                "context": [],
                "sources": []
            }

    async def _create_visualization(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Create data visualization"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Extract data from content
            data = self._extract_data_for_visualization(content)
            
            # Create visualization based on data type
            if isinstance(data, pd.DataFrame):
                sns.set_style("whitegrid")
                self._create_data_visualization(data, content.get("visualization_type", "line"))
            else:
                # Default to bar chart for simple data
                plt.bar(range(len(data)), list(data.values()), align='center')
                plt.xticks(range(len(data)), list(data.keys()), rotation=45)

            plt.title(content.get("title", "Visualization"))
            plt.tight_layout()
            
            # Save to bytes and convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
            plt.close()
            buffer.seek(0)
            
            return {
                "content": base64.b64encode(buffer.getvalue()).decode(),
                "format": "png",
                "title": content.get("title", "Visualization")
            }
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return {"error": str(e)}

    def _create_data_visualization(self, data: pd.DataFrame, viz_type: str) -> None:
        """Create specific type of visualization"""
        if viz_type == "line":
            sns.lineplot(data=data)
        elif viz_type == "bar":
            sns.barplot(data=data)
        elif viz_type == "scatter":
            sns.scatterplot(data=data)
        elif viz_type == "heatmap":
            sns.heatmap(data, annot=True, cmap="YlOrRd")
        else:
            # Default to line plot
            sns.lineplot(data=data)

    def _extract_data_for_visualization(self, content: Dict[str, Any]) -> Any:
        """Extract and format data for visualization"""
        data = {}
        
        # Try to find numerical data in content
        for section in content.get("sections", []):
            if section.get("data"):
                return pd.DataFrame(section["data"])
            
            # Extract numbers from content
            for line in section.get("content", []):
                numbers = re.findall(r'\d+(?:\.\d+)?', line)
                if numbers:
                    data[section["title"]] = float(numbers[0])
                    
        return data or {"No Data": 0}

    def _structure_content(self, content: str) -> Dict[str, Any]:
        """Structure the generated content"""
        structured_content = {
            "title": "",
            "subtitle": "",
            "sections": [],
            "metadata": {
                "type": self.content_type,
                "timestamp": datetime.now().isoformat()
            }
        }

        current_section = None
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue

            if line.startswith('# '):
                structured_content["title"] = line.lstrip('# ').strip()
            elif line.startswith('## '):
                if current_section:
                    structured_content["sections"].append(current_section)
                current_section = {
                    "title": line.lstrip('## ').strip(),
                    "content": [],
                    "bullets": []
                }
            elif line.startswith('- ') and current_section:
                current_section["bullets"].append(line.lstrip('- ').strip())
            elif current_section:
                current_section["content"].append(line)

        if current_section:
            structured_content["sections"].append(current_section)

        return structured_content

    async def _create_infographic(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Create an infographic visualization"""
        try:
            # Create a figure with custom dimensions for infographic
            plt.figure(figsize=(12, 18))
            
            # Extract key information
            title = content.get("title", "Infographic")
            sections = content.get("sections", [])
            
            # Setup the layout
            gs = plt.GridSpec(len(sections) + 1, 1, height_ratios=[1] + [2] * len(sections))
            
            # Add title
            plt.subplot(gs[0])
            plt.text(0.5, 0.5, title, 
                    ha='center', va='center', 
                    fontsize=20, fontweight='bold')
            plt.axis('off')
            
            # Add sections
            for i, section in enumerate(sections, 1):
                plt.subplot(gs[i])
                self._add_infographic_section(section, i, len(sections))
            
            plt.tight_layout()
            
            # Save to bytes and convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
            plt.close()
            buffer.seek(0)
            
            return {
                "content": base64.b64encode(buffer.getvalue()).decode(),
                "format": "png",
                "title": title
            }
            
        except Exception as e:
            logger.error(f"Error creating infographic: {e}")
            return {"error": str(e)}

    async def _create_report(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Create a detailed report"""
        try:
            doc = Document()
            
            # Add title page
            doc.add_heading(content["title"], 0)
            doc.add_paragraph().add_run("Generated Report").bold = True
            doc.add_paragraph().add_run(f"Date: {datetime.now().strftime('%Y-%m-%d')}").italic = True
            
            # Add table of contents
            doc.add_page_break()
            doc.add_heading("Table of Contents", 1)
            toc = doc.add_paragraph()
            for section in content.get("sections", []):
                toc.add_run(f"{section['title']}\n")
            
            # Add executive summary
            doc.add_page_break()
            doc.add_heading("Executive Summary", 1)
            doc.add_paragraph("\n".join(content.get("sections", [{}])[0].get("content", [])))
            
            # Add main content
            for section in content.get("sections", [])[1:]:
                doc.add_page_break()
                doc.add_heading(section["title"], 1)
                
                if section.get("content"):
                    doc.add_paragraph("\n".join(section["content"]))
                    
                if section.get("bullets"):
                    for bullet in section["bullets"]:
                        doc.add_paragraph(bullet, style='List Bullet')
            
            # Save to bytes and convert to base64
            docx_buffer = io.BytesIO()
            doc.save(docx_buffer)
            docx_buffer.seek(0)
            
            return {
                "content": base64.b64encode(docx_buffer.getvalue()).decode(),
                "format": "docx",
                "title": content.get("title", "Report")
            }
            
        except Exception as e:
            logger.error(f"Error creating report: {e}")
            return {"error": str(e)}

    async def _create_email(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Create email content"""
        try:
            email_parts = []
            
            # Add subject
            subject = content.get("title", "").strip()
            
            # Add greeting
            if content.get("sections"):
                first_section = content["sections"][0]
                greeting = first_section.get("content", [""])[0]
                email_parts.append(greeting)
            
            # Add body
            for section in content.get("sections", [])[1:]:
                # Add section content
                if section.get("content"):
                    email_parts.extend(section["content"])
                
                # Add bullets if any
                if section.get("bullets"):
                    email_parts.append("")  # Add spacing
                    email_parts.extend([f"• {bullet}" for bullet in section["bullets"]])
                
                email_parts.append("")  # Add spacing between sections
            
            # Add signature
            email_parts.append("\nBest regards,")
            email_parts.append("[Your name]")
            
            return {
                "subject": subject,
                "body": "\n".join(email_parts),
                "format": "email"
            }
            
        except Exception as e:
            logger.error(f"Error creating email: {e}")
            return {"error": str(e)}

    async def _create_social_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Create social media content"""
        try:
            social_content = {
                "title": content.get("title", ""),
                "main_content": "",
                "hashtags": [],
                "format": "social"
            }
            
            # Extract hashtags from content
            hashtags = set()
            for section in content.get("sections", []):
                words = " ".join(section.get("content", []) + section.get("bullets", [])).split()
                hashtags.update([word for word in words if word.startswith("#")])
                
            # Create main content
            main_parts = []
            for section in content.get("sections", []):
                if section.get("content"):
                    main_parts.extend(section["content"])
                if section.get("bullets"):
                    main_parts.extend([f"• {bullet}" for bullet in section["bullets"]])
            
            social_content["main_content"] = "\n".join(main_parts)
            social_content["hashtags"] = list(hashtags)
            
            return social_content
            
        except Exception as e:
            logger.error(f"Error creating social content: {e}")
            return {"error": str(e)}

    def _add_infographic_section(self, section: Dict[str, Any], position: int, total_sections: int):
        """Add a section to the infographic"""
        plt.text(0.5, 0.9, section["title"], 
                ha='center', va='top', 
                fontsize=16, fontweight='bold')
        
        # Add content
        content_text = "\n".join(section.get("content", []))
        plt.text(0.1, 0.7, content_text,
                ha='left', va='top',
                fontsize=12)
        
        # Add bullets
        bullet_text = "\n".join([f"• {bullet}" for bullet in section.get("bullets", [])])
        plt.text(0.1, 0.4, bullet_text,
                ha='left', va='top',
                fontsize=12)
        
        plt.axis('off')
        
    async def _create_summary(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Create a concise summary of the content"""
        try:
            # Extract main points
            sections = content.get("sections", [])
            title = content.get("title", "")
            summary_parts = [title]
            
            # Add key points from each section
            for section in sections:
                section_title = section.get("title", "")
                if section_title:
                    summary_parts.append(section_title)
                    
                # Add bullets as points
                bullets = section.get("bullets", [])
                if bullets:
                    summary_parts.extend(f"• {bullet}" for bullet in bullets[:3])  # Limit to top 3 bullets
                    
                # Add first line of content if available
                if section.get("content"):
                    first_line = section["content"][0].strip()
                    if first_line and len(first_line) > 10:  # Only meaningful content
                        summary_parts.append(first_line)
            
            return {
                "content": "\n\n".join(summary_parts),
                "format": "text",
                "title": title
            }
                
        except Exception as e:
            logger.error(f"Error creating summary: {e}")
            return {"error": str(e)}