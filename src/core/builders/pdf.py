"""
Contains the PDF model
"""
import logging

from helpers.utils import txt_to_pdf
from core.builders.base_builder import BaseBuilder
from core.functions import categories_to_txt


log = logging.getLogger("app")



class PDF(BaseBuilder):
    """
    PDF model.
    
    Builds category PDFs from outputs.
    """
    def build(self, stage_name: str) -> None:
        """
        Builds the final form Category PDF files for Stage 1, 1r, & 1c.

        Saves PDF to subpath.
        """
        # Build PDF content.
        pdf_content = self._generate_pdf_content(stage_name)
        pdf_path = self.sub_path / self.file_names["categories"][stage_name]
        
        # Saving PDF.
        txt_to_pdf(pdf_content, pdf_path)
        log.info(f"Stage {stage_name} PDF saved to: {pdf_path}")

    def _generate_pdf_content(self, stage_name: str) -> str:
        """
        Generates the content for the PDF.
        """
        content = [f"# Stage {stage_name} Categories\n"]
        outputs = self.stage_outputs[stage_name]
        
        for subset, output in outputs.outputs.items():
            categories = self.output_attrb(output[0].request_out.parsed)
            
            if subset != "full":
                case, instance_type = subset.split("_")
                content.append(f"## {case.capitalize()} - {instance_type.upper()} Categories\n")
            else:
                if stage_name == "1c":
                    content.append("## Final Category Set\n")
                else:
                    content.append("## Unified Categories\n")
            
            content.append(categories_to_txt(categories))
        
        return "\n".join(content)