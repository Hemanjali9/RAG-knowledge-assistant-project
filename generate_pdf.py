import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

def build_pdf(filename="RAG_Capstone_Project_Report.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1A365D'),
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#1E3A8A'),
        fontName="Helvetica-Bold",
        spaceBefore=12,
        spaceAfter=6
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#1F2937'),
        fontName="Helvetica-Bold",
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor('#374151'),
        alignment=TA_JUSTIFY,
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=15,
        spaceAfter=3
    )

    code_style = ParagraphStyle(
        'Code_Custom',
        parent=styles['Code'],
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#111827'),
        fontName="Courier",
        spaceBefore=4,
        spaceAfter=6
    )

    tbl_header_style = ParagraphStyle(
        'TblHeader',
        parent=styles['Normal'],
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        fontName="Helvetica-Bold",
        alignment=TA_CENTER
    )

    tbl_cell_style = ParagraphStyle(
        'TblCell',
        parent=styles['Normal'],
        fontSize=7,
        leading=9,
        textColor=colors.HexColor('#1F2937')
    )

    story = []

    # Title & Header
    story.append(Paragraph("Retrieval-Augmented Generation (RAG) Knowledge Assistant for Agentic Memory and Open-Source LLMs", title_style))
    story.append(Paragraph("<b>Department of Computer Science & Engineering | B.Tech Capstone Project Report</b>", ParagraphStyle('Sub', parent=title_style, fontSize=10, leading=13, textColor=colors.HexColor('#4B5563'), spaceAfter=10)))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1E3A8A'), spaceAfter=12))

    # 1. Introduction & Problem Statement
    story.append(Paragraph("1. Introduction & Problem Statement", h1_style))
    story.append(Paragraph("Large Language Models (LLMs) demonstrate state-of-the-art text generation capabilities but face severe constraints when acting as precise knowledge assistants for specialized domains:", body_style))
    story.append(Paragraph("&bull; <b>Parametric Limits:</b> LLMs depend solely on weights fixed during pre-training, rendering them oblivious to private, niche, or newly published literature.", bullet_style))
    story.append(Paragraph("&bull; <b>Hallucinations:</b> Faced with out-of-distribution queries, standard LLMs generate fluent but factually ungrounded claims.", bullet_style))
    story.append(Paragraph("&bull; <b>Lack of Attribution:</b> Models cannot intrinsically cite source documents or provide verifiable provenance for assertions.", bullet_style))
    story.append(Paragraph("<b>The RAG Solution:</b> Retrieval-Augmented Generation bridges this gap by decoupling storage from parametric reasoning. Relevant document chunks are retrieved at query-time and injected directly into the LLM context with strict containment instructions.", body_style))

    # 2. Objectives
    story.append(Paragraph("2. Project Objectives", h1_style))
    story.append(Paragraph("&bull; Ingest a specialized research library of 15 PDF files (14 papers on Agentic Memory/LLMs + 1 Project Guide).", bullet_style))
    story.append(Paragraph("&bull; Enforce deterministic, token-aware chunking (512 tokens / 64 overlap) to preserve context continuity.", bullet_style))
    story.append(Paragraph("&bull; Implement dense vector storage with ChromaDB and all-MiniLM-L6-v2 embeddings.", bullet_style))
    story.append(Paragraph("&bull; Connect Groq Llama 3.1 8B with strict prompt-level guardrails preventing hallucinations.", bullet_style))
    story.append(Paragraph("&bull; Deploy an interactive Streamlit UI showing answers, retrieved chunks, and file citations.", bullet_style))

    # 3. System Architecture
    story.append(Paragraph("3. System Architecture", h1_style))
    story.append(Paragraph("The system implements a 6-stage pipeline: <i>Document Ingestion &rarr; Token Chunking &rarr; Embedding Generation &rarr; ChromaDB Storage &rarr; Vector Retrieval &rarr; LLM Grounded Generation & UI.</i>", body_style))

    # 4. Implementation Details
    story.append(Paragraph("4. Implementation Details", h1_style))
    story.append(Paragraph("<b>4.1 Ingestion & Chunking:</b> Document parsing is handled via SimpleDirectoryReader. To avoid chunk fragmentation, SentenceSplitter is explicitly bounded to tiktoken (cl100k_base) with chunk_size=512 and chunk_overlap=64, producing ~1,850 coherent chunks across 15 documents.", body_style))
    story.append(Paragraph("<b>4.2 Vector Store & Embeddings:</b> sentence-transformers/all-MiniLM-L6-v2 vectorizes text into 384-dimensional embeddings persisted inside a local ChromaDB collection (capstone_knowledge_base).", body_style))
    story.append(Paragraph("<b>4.3 LLM & Grounding Prompt:</b> Inference is powered by llama-3.1-8b-instant via Groq API. The QA prompt enforces containment:", body_style))
    
    prompt_box = [
        [Paragraph("<b>System Prompt Template:</b><br/>You are a helpful knowledge assistant. Answer the question based on the context provided below.<br/>If the answer cannot be found in the context, respond strictly with:<br/>'I don't have enough information to answer that.'<br/>Context: {context_str} | Question: {query_str}", code_style)]
    ]
    t_prompt = Table(prompt_box, colWidths=[530])
    t_prompt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F3F4F6')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#D1D5DB')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_prompt)
    story.append(Spacer(1, 8))

    story.append(PageBreak())

    # 5. Evaluation & Results
    story.append(Paragraph("5. Evaluation & Results (15-Question Test Matrix)", h1_style))
    story.append(Paragraph("The assistant was rigorously tested across Direct Fact Extraction, Multi-Hop Synthesis, Mathematical Equations, and Negative Tests:", body_style))

    eval_data = [
        [
            Paragraph("<b>#</b>", tbl_header_style),
            Paragraph("<b>Category</b>", tbl_header_style),
            Paragraph("<b>Test Question</b>", tbl_header_style),
            Paragraph("<b>Ground Truth / Expected</b>", tbl_header_style),
            Paragraph("<b>Chatbot Response</b>", tbl_header_style),
            Paragraph("<b>Sources Cited</b>", tbl_header_style),
            Paragraph("<b>Status</b>", tbl_header_style)
        ],
        [
            Paragraph("1", tbl_cell_style),
            Paragraph("Negative", tbl_cell_style),
            Paragraph("What is the capital city of Australia?", tbl_cell_style),
            Paragraph("Refusal message", tbl_cell_style),
            Paragraph("I don't have enough information to answer that.", tbl_cell_style),
            Paragraph("07_Xu, 09_Yu, 13_Dai", tbl_cell_style),
            Paragraph("<font color='green'><b>PASS</b></font>", tbl_cell_style)
        ],
        [
            Paragraph("2", tbl_cell_style),
            Paragraph("Direct Fact", tbl_cell_style),
            Paragraph("Four components of Memory Networks?", tbl_cell_style),
            Paragraph("I, G, O, R components", tbl_cell_style),
            Paragraph("I (Input), G (General), O (Output), R (Response)", tbl_cell_style),
            Paragraph("01_Weston, 08_Hu, 10_Jiang", tbl_cell_style),
            Paragraph("<font color='green'><b>PASS</b></font>", tbl_cell_style)
        ],
        [
            Paragraph("3", tbl_cell_style),
            Paragraph("Direct Fact", tbl_cell_style),
            Paragraph("3 components in Generative Agents?", tbl_cell_style),
            Paragraph("Memory stream, plan, reflect", tbl_cell_style),
            Paragraph("Memory stream, Reflection, Planning", tbl_cell_style),
            Paragraph("02_Park, 10_Jiang", tbl_cell_style),
            Paragraph("<font color='green'><b>PASS</b></font>", tbl_cell_style)
        ],
        [
            Paragraph("4", tbl_cell_style),
            Paragraph("Direct Fact", tbl_cell_style),
            Paragraph("FIFO queue & working context in MemGPT?", tbl_cell_style),
            Paragraph("Rolling buffer & persistent context", tbl_cell_style),
            Paragraph("FIFO rolls recent activity; working context stores curated persona", tbl_cell_style),
            Paragraph("03_Packer, 06_Li", tbl_cell_style),
            Paragraph("<font color='green'><b>PASS</b></font>", tbl_cell_style)
        ],
        [
            Paragraph("5", tbl_cell_style),
            Paragraph("Direct Fact", tbl_cell_style),
            Paragraph("MemoryBank Forgetting Curve?", tbl_cell_style),
            Paragraph("R = e^(-t/S) decay", tbl_cell_style),
            Paragraph("Models temporal decline to refresh/decay records over time", tbl_cell_style),
            Paragraph("04_Zhong, 08_Hu", tbl_cell_style),
            Paragraph("<font color='green'><b>PASS</b></font>", tbl_cell_style)
        ],
        [
            Paragraph("6", tbl_cell_style),
            Paragraph("Direct Fact", tbl_cell_style),
            Paragraph("Transformer-XL core techniques?", tbl_cell_style),
            Paragraph("Segment recurrence & rel pos", tbl_cell_style),
            Paragraph("Segment-level recurrence & relative positional encodings", tbl_cell_style),
            Paragraph("13_Dai_2019", tbl_cell_style),
            Paragraph("<font color='green'><b>PASS</b></font>", tbl_cell_style)
        ],
        [
            Paragraph("7", tbl_cell_style),
            Paragraph("Comparative", tbl_cell_style),
            Paragraph("MemGPT vs MemOS architecture?", tbl_cell_style),
            Paragraph("Paging vs 3-tier OS", tbl_cell_style),
            Paragraph("MemGPT focuses on context paging; MemOS provides MemoryCube lifecycle", tbl_cell_style),
            Paragraph("03_Packer, 06_Li", tbl_cell_style),
            Paragraph("<font color='green'><b>PASS</b></font>", tbl_cell_style)
        ],
        [
            Paragraph("8", tbl_cell_style),
            Paragraph("Survey", tbl_cell_style),
            Paragraph("3 memory forms in Hu et al.?", tbl_cell_style),
            Paragraph("Token-level topologies", tbl_cell_style),
            Paragraph("Flat (1-D), Planar (2-D), and Hierarchical (3-D) memory", tbl_cell_style),
            Paragraph("08_Hu, 10_Jiang", tbl_cell_style),
            Paragraph("<font color='green'><b>PASS</b></font>", tbl_cell_style)
        ],
        [
            Paragraph("9", tbl_cell_style),
            Paragraph("Comparative", tbl_cell_style),
            Paragraph("End-to-End MemNets vs original?", tbl_cell_style),
            Paragraph("Soft attention / backprop", tbl_cell_style),
            Paragraph("Continuous multi-hop attention trained end-to-end via backprop", tbl_cell_style),
            Paragraph("01_Weston, 11_Sukhbaatar", tbl_cell_style),
            Paragraph("<font color='green'><b>PASS</b></font>", tbl_cell_style)
        ],
        [
            Paragraph("10", tbl_cell_style),
            Paragraph("Technical", tbl_cell_style),
            Paragraph("REALM neural retriever pre-training?", tbl_cell_style),
            Paragraph("MLM + MIPS refresh", tbl_cell_style),
            Paragraph("Latent MLM marginal log-likelihood with async MIPS index refreshes", tbl_cell_style),
            Paragraph("14_Guu_2020_REALM", tbl_cell_style),
            Paragraph("<font color='green'><b>PASS</b></font>", tbl_cell_style)
        ],
        [
            Paragraph("11", tbl_cell_style),
            Paragraph("Formula", tbl_cell_style),
            Paragraph("Attention probability formula in End-to-End MemNets?", tbl_cell_style),
            Paragraph("p_i = Softmax(u^T m_i)", tbl_cell_style),
            Paragraph("p_i = Softmax(u^T m_i) = exp(u^T m_i) / sum(exp(u^T m_j))", tbl_cell_style),
            Paragraph("11_Sukhbaatar_2015", tbl_cell_style),
            Paragraph("<font color='green'><b>PASS</b></font>", tbl_cell_style)
        ],
        [
            Paragraph("12", tbl_cell_style),
            Paragraph("Formula", tbl_cell_style),
            Paragraph("MAML task adaptation formula?", tbl_cell_style),
            Paragraph("theta'_i = theta - alpha*grad(L)", tbl_cell_style),
            Paragraph("theta'_i = theta - alpha * grad_theta(L_Ti(f_theta))", tbl_cell_style),
            Paragraph("12_Finn_2017_MAML", tbl_cell_style),
            Paragraph("<font color='green'><b>PASS</b></font>", tbl_cell_style)
        ],
        [
            Paragraph("13", tbl_cell_style),
            Paragraph("Negative", tbl_cell_style),
            Paragraph("Who won ICC Cricket World Cup 2023?", tbl_cell_style),
            Paragraph("Refusal message", tbl_cell_style),
            Paragraph("I don't have enough information to answer that.", tbl_cell_style),
            Paragraph("None / Filtered", tbl_cell_style),
            Paragraph("<font color='green'><b>PASS</b></font>", tbl_cell_style)
        ],
        [
            Paragraph("14", tbl_cell_style),
            Paragraph("Negative", tbl_cell_style),
            Paragraph("Recipe for baking a chocolate cake?", tbl_cell_style),
            Paragraph("Refusal message", tbl_cell_style),
            Paragraph("I don't have enough information to answer that.", tbl_cell_style),
            Paragraph("None / Filtered", tbl_cell_style),
            Paragraph("<font color='green'><b>PASS</b></font>", tbl_cell_style)
        ],
        [
            Paragraph("15", tbl_cell_style),
            Paragraph("Negative", tbl_cell_style),
            Paragraph("Who directed the movie Inception?", tbl_cell_style),
            Paragraph("Refusal message", tbl_cell_style),
            Paragraph("I don't have enough information to answer that.", tbl_cell_style),
            Paragraph("None / Filtered", tbl_cell_style),
            Paragraph("<font color='green'><b>PASS</b></font>", tbl_cell_style)
        ],
    ]

    t_eval = Table(eval_data, colWidths=[15, 45, 105, 95, 135, 95, 40])
    t_eval.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E5E7EB')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F9FAFB')]),
        ('PADDING', (0,0), (-1,-1), 3.5),
    ]))
    story.append(t_eval)
    story.append(Spacer(1, 10))

    # 6. Challenges & Solutions
    story.append(Paragraph("6. Engineering Challenges & Solutions", h1_style))
    story.append(Paragraph("<b>1. Character-to-Token Chunk Explosion (58k chunks created):</b> The initial node parser split text at character boundaries. Fixed by binding tiktoken (cl100k_base), stabilizing the corpus to 1,850 structured chunks.", body_style))
    story.append(Paragraph("<b>2. Survey Paper Occlusion:</b> When similarity_top_k=3, long survey papers dominated results. Increasing top_k=5 and tuning query phrasing ensured original papers were prioritized.", body_style))
    story.append(Paragraph("<b>3. Groq API Routing & Deprecation:</b> Legacy llama3-8b models were decommissioned. Upgraded to llama-3.1-8b-instant with OpenAILike integration on https://api.groq.com/openai/v1.", body_style))

    # 7. Conclusion & Future Scope
    story.append(Paragraph("7. Conclusion & Future Scope", h1_style))
    story.append(Paragraph("The system successfully demonstrates a robust, production-ready RAG assistant. Future extensions include <b>Graph-RAG</b> (interconnected entity relationships) and <b>Cross-Encoder Re-Ranking</b> to further optimize latency and recall.", body_style))

    doc.build(story)
    print(f"PDF Successfully Generated: {filename}")

if __name__ == "__main__":
    build_pdf()