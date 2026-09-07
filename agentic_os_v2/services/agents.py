import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from services.aws_bedrock import bedrock_client
from services.llm_provider import llm_provider
from core.config import LEAD_MODEL_ID, SPECIALIST_MODEL_ID
import core.database as db
from services.planner import generate_day_schedule
from services.document_agent import document_agent
from services.ntulearn import ntulearn_service

LEAD_SYSTEM_PROMPT = """You are the Lead Orchestrator of CogniFlow OS.
You supervise multiple specialized sub-agents to organize the student's academic life for Semester 1 (26S1).

Available Sub-Agents to Delegate to:
1. `task_document_specialist`: Inspects course slides, searches tutorial documents, extracts semester schedules, and retrieves specific PDF materials or test/exam information. Pass both `course_code` (e.g. 'MH2500', 'SC2001', 'SC2207', 'MH2802', 'CC0006') and `detail` (e.g. 'schedule', 'test', 'mock', 'Tutorial 1', 'syllabus') so it can deeply search document contents, questions, solutions, and milestones.
2. `task_planner_agent`: Generates optimized time-blocked study schedules and calculates urgency priority scores.
3. `task_ntulearn_specialist`: Checks and reads the full text bodies of announcements, deadlines, test venues, and course notices.
4. `create_action_task`: Creates and logs a new actionable task, reminder, interview, deadline, or study commitment into the student's Prioritized Action Items list and SQLite database. Whenever the student asks to add a task, schedule an interview, or set an action item, you MUST execute this tool with a descriptive title, due date/time, and priority score so it actually gets saved.

Guidelines:
- **Course Accuracy & Broad Context**: Accurately identify the exact course code requested by the student (e.g. CC0006, MH2500, MH2802, ML0004, SC2001, SC2207). If the student mentions a course abbreviation (e.g. 'cc6' -> CC0006, 'sc2207' -> SC2207, 'ml4' -> ML0004), resolve it to the exact course. NEVER assume or fixate on any single module. If the student asks a general question (e.g., general study plan, urgent deadlines across all courses, daily schedule, greeting), address all relevant modules across their academic workload rather than limiting your answer to one course.
- **Sub-Agent Delegation**: Whenever the student asks "what do I need to do", "which tutorial do I need", "find tutorial", or asks about course content/schedules/tests, ALWAYS task `task_document_specialist` and/or `task_ntulearn_specialist` to inspect the actual course documents and notices first.
- **Announcements & Notices**: When asked about test venues, deadlines, or schedules, ALWAYS delegate to `task_ntulearn_specialist` and check the announcement notices and bodies. Extract and quote the actual dates, venues, and instructions!
- **Document Citing**: Clearly distinguish between Question sheets, Solution hints, Mock test papers, and Syllabus documents. ALWAYS include the clickable action token:
  `[OPEN_DOC:<material_id>:<Document Title>]`
  This allows the student to click directly in the chat to open and view the PDF in the application!
- **Response Structure**:
  - 🎯 **Summary & Immediate Priorities**
  - 📄 **Exact Tutorial / Test / Syllabus PDFs Required & Action Links**
  - 📅 **Semester Schedule & Milestones**
  - 💡 **Recommended Next Step**
- **Temporal Precision & Day of Week**:
  Today is strictly **Monday, 07 September 2026** (e.g. 7 September 2026 is **Monday**, NOT Saturday). Never assume or state an incorrect day of the week. Calculate all deadlines, tutorial dates, and class schedules relative to today being Monday.
- **Dedicated Spoken Voice Summary (MANDATORY)**:
  At the very end of your final response, append a dedicated voice summary block formatted EXACTLY as:
  `[VOICE_SUMMARY: 3-4 complete, conversational spoken sentences (roughly 45-75 words). Provide a natural, thorough spoken summary. Never cut off abruptly; always finish your complete sentences with closing punctuation. Strictly zero emojis, zero markdown, zero bullet points.]`
  Example:
  [VOICE_SUMMARY: You have an MH2500 lecture at 9:30 AM and an SC2001 tutorial at 2:30 PM today.]
- Professional, encouraging, and clear executive tone.
"""

TOOL_DEFINITIONS = [
    {
        "toolSpec": {
            "name": "task_document_specialist",
            "description": "Task the Document Specialist Sub-Agent to deeply inspect, analyze, or search course documents, lecture slides, syllabus timetables, and tutorial question/solution sheets for a course. Searches inside document text, extracted questions, and rosters.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "course_code": {
                            "type": "string",
                            "description": "Course code or keyword, e.g. 'MH2500', 'SC2001', 'SC2207', 'MH2802', 'CC0006', 'ML0004'"
                        },
                        "task_type": {
                            "type": "string",
                            "enum": ["find_tutorial_pdf", "extract_schedule", "inspect_lecture_slides", "get_solutions", "general_search"],
                            "description": "Specific intent for the document agent"
                        },
                        "detail": {
                            "type": "string",
                            "description": "Topic, keyword, or specific tutorial/test name to search within the text/questions, e.g. 'schedule', 'test', 'mock', 'Tutorial 1', 'probability'"
                        }
                    },
                    "required": ["task_type"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "task_planner_agent",
            "description": "Task the Planner Sub-Agent to calculate priority rankings, deadlines, and dynamic calendar time-blocking.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "available_hours": {
                            "type": "number",
                            "description": "Number of available study hours, e.g. 3.0, 6.0"
                        },
                        "focus_module": {
                            "type": "string",
                            "description": "Optional specific course code to prioritize"
                        }
                    }
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "task_ntulearn_specialist",
            "description": "Task the NTULearn Specialist Sub-Agent to inspect announcements, deadlines, test venues, and module notices from Blackboard Ultra.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "course_code": {
                            "type": "string",
                            "description": "Course code to filter announcements"
                        },
                        "query": {
                            "type": "string",
                            "description": "Search topic e.g. 'quiz', 'test', 'venue', 'exam', 'slot', 'schedule'"
                        }
                    }
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "create_action_task",
            "description": "Create and log a new actionable task, reminder, interview, deadline, or study commitment into the student's Prioritized Action Items list and SQLite database.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Clear title of the task (e.g. 'Interview (4:00 PM)', 'Review BS1016 Lecture 3')"
                        },
                        "course_code": {
                            "type": "string",
                            "description": "Course code or 'General' for personal/extracurricular events"
                        },
                        "due_date": {
                            "type": "string",
                            "description": "Due date/time (e.g. '2026-09-07 16:00' or 'Today 4:00 PM')"
                        },
                        "estimated_minutes": {
                            "type": "integer",
                            "description": "Estimated duration in minutes (e.g. 60)"
                        },
                        "priority_score": {
                            "type": "number",
                            "description": "Priority score between 1.0 and 10.0 (e.g. 9.5 for urgent commitments/interviews)"
                        },
                        "notes": {
                            "type": "string",
                            "description": "Optional notes or details"
                        }
                    },
                    "required": ["title"]
                }
            }
        }
    }
]

# Convert Bedrock toolSpec format to standard OpenAI/Groq/Gemini tool format
OPENAI_TOOL_DEFINITIONS = []
for t in TOOL_DEFINITIONS:
    if "toolSpec" in t:
        spec = t["toolSpec"]
        OPENAI_TOOL_DEFINITIONS.append({
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec.get("inputSchema", {}).get("json", {})
            }
        })


def normalize_course_token(token: str) -> Optional[str]:
    """
    Normalizes course abbreviations like 'cc6', 'cc06', 'cc0006', 'sc2207', 'ml4', 'mh2500'.
    """
    if not token:
        return None
    token = token.strip().upper()
    m = re.match(r'^([A-Z]{2,4})[\s-]*0*(\d{1,4})([A-Z]?)$', token)
    if m:
        prefix = m.group(1)
        num = int(m.group(2))
        suffix = m.group(3)
        return f"{prefix}{num:04d}{suffix}"
    return None


def extract_course_from_query(query: str, history: List[Dict[str, Any]], courses: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Detects target course from query, subject keywords, or multi-turn history with high precision."""
    if not courses or not query:
        return None

    clean_map = {}
    for c in courses:
        raw_code = c.get("course_code", "").upper()
        clean_map[raw_code] = c
        m = re.search(r'\b([A-Z]{2,4}\d{4}[A-Z]?)\b', raw_code)
        if m:
            clean_map[m.group(1)] = c

    # 1. Direct course code regex match in user query (handles full codes and abbreviations like cc6, ml4, sc2001)
    for m in re.finditer(r'\b([A-Za-z]{2,4}[\s-]*\d{1,4}[A-Za-z]?)\b', query):
        norm = normalize_course_token(m.group(0))
        if norm and norm in clean_map:
            return clean_map[norm]

    # 2. Match subject keywords in user query (sorted by length descending to match multi-word terms first)
    keyword_map = {
        # CC0006
        "sustainability": "CC0006",
        "sustainable": "CC0006",
        "climate": "CC0006",
        "carbon": "CC0006",
        "green": "CC0006",
        "society economy": "CC0006",
        # MH2500
        "probability": "MH2500",
        "prob": "MH2500",
        "statistics": "MH2500",
        "stats": "MH2500",
        "random variable": "MH2500",
        "combinatorics": "MH2500",
        "combinatorial": "MH2500",
        "distribution": "MH2500",
        "bayes": "MH2500",
        # MH2802
        "linear algebra": "MH2802",
        "linalg": "MH2802",
        "matrix": "MH2802",
        "matrices": "MH2802",
        "eigenvalue": "MH2802",
        "eigenvector": "MH2802",
        "vector space": "MH2802",
        # SC2001
        "algorithm": "SC2001",
        "algorithms": "SC2001",
        "algo": "SC2001",
        "dynamic programming": "SC2001",
        "dijkstra": "SC2001",
        "graph": "SC2001",
        "sorting": "SC2001",
        "complexity": "SC2001",
        # SC2207
        "database": "SC2207",
        "databases": "SC2207",
        "db": "SC2207",
        "sql": "SC2207",
        "normalization": "SC2207",
        "relational": "SC2207",
        "er diagram": "SC2207",
        # ML0004
        "career": "ML0004",
        "careers": "ML0004",
        "internship": "ML0004",
        "resume": "ML0004",
        "entrepreneur": "ML0004",
        "entrepreneurship": "ML0004",
    }

    q_lower = query.lower()
    for kw, code in sorted(keyword_map.items(), key=lambda x: -len(x[0])):
        if re.search(r'\b' + re.escape(kw) + r'\b', q_lower):
            if code in clean_map:
                return clean_map[code]

    # 3. Check if query is a greeting or general cross-module prompt (NEVER latch onto past course)
    q_clean = query.strip().lower()
    if re.search(r'^(hello|hi|hey|good\s*(morning|afternoon|evening)|who\s*are\s*you|howdy|sup)\b', q_clean):
        return None

    general_patterns = [
        r'\b(all\s+courses|all\s+modules|everything|overall|semester)\b',
        r'\b(3-hour|3\s+hour|study\s+plan|plan\s+my\s+day|tonight)\b',
        r'\b(urgent\s+deadlines?|deadlines?|any\s+tests?|upcoming\s+tests?)\b',
        r'\b(what\s+do\s+i\s+need\s+to\s+do|what\s+tutorial\s+do\s+i\s+need|which\s+tutorial)\b',
        r'\b(today[\'\s]s\s+schedule|my\s+timetable|today\s+schedule)\b',
    ]
    if any(re.search(pat, q_clean) for pat in general_patterns):
        return None

    # 4. Contextual follow-up only: check if user query references a previous topic using pronouns/follow-up markers
    is_followup = bool(re.search(r'\b(it|its|that|this|the\s+test|the\s+exam|the\s+quiz|the\s+assignment|the\s+lab|the\s+lecture|more\s+details|venue|solutions?)\b', q_clean))
    if is_followup and history:
        # Check only the immediate 1-2 turns back; never crawl back across ancient messages!
        recent = [h for h in history if h.get("content")][-2:]
        for h in reversed(recent):
            txt = h.get("content", "")
            for m in re.finditer(r'\b([A-Za-z]{2,4}[\s-]*\d{1,4}[A-Za-z]?)\b', txt):
                norm = normalize_course_token(m.group(0))
                if norm and norm in clean_map:
                    return clean_map[norm]
            for kw, code in sorted(keyword_map.items(), key=lambda x: -len(x[0])):
                if re.search(r'\b' + re.escape(kw) + r'\b', txt.lower()) and code in clean_map:
                    return clean_map[code]

    return None

def execute_subagent_tool(name: str, args: Dict[str, Any], term: str = "26S1") -> Dict[str, Any]:
    """Dispatches tool call directly to the requested specialist sub-agent."""
    if name == "task_document_specialist":
        raw_code = args.get("course_code", "")
        task_type = args.get("task_type", "general_search")
        detail = args.get("detail", "")

        # Clean & normalize course code (e.g. MH2500 from 26S1-MH2500-LEC or cc6 -> CC0006)
        norm_c = normalize_course_token(raw_code)
        cm = re.search(r'\b([A-Z]{2,4}\d{4}[A-Z]?)\b', raw_code, re.IGNORECASE)
        course_code = norm_c if norm_c else (cm.group(1).upper() if cm else raw_code)

        matching_mats = []
        if detail and hasattr(db, "search_documents_by_content"):
            matching_mats = db.search_documents_by_content(query=detail, course_code=course_code)

        combined_text = f"{task_type} {detail}".lower()
        is_test_query = any(w in combined_text for w in ["test", "exam", "quiz", "mock", "ca1", "ca2", "assessment"])
        is_syllabus_query = any(w in combined_text for w in ["syllabus", "schedule", "timetable", "outline", "hand00", "overview"])

        all_mats = db.get_all_materials(course_code=course_code)

        if (is_test_query or is_syllabus_query) and not matching_mats:
            syllabus_mats = [
                m for m in all_mats
                if any(w in m.get("title", "").lower() for w in ["hand00", "syllabus", "schedule", "outline", "overview"])
            ]
            test_mats = [
                m for m in all_mats
                if any(w in m.get("title", "").lower() for w in ["test", "mock", "exam", "quiz", "ca1", "ca2"])
            ]
            if is_test_query and is_syllabus_query:
                matching_mats = syllabus_mats + test_mats
            elif is_test_query:
                matching_mats = test_mats + syllabus_mats
            else:
                matching_mats = syllabus_mats + test_mats

        # Fall back to matching tutorials if needed
        if not matching_mats:
            matching_mats = db.find_matching_tutorials(course_code=course_code, query=detail)

        # Fall back to general course materials
        if not matching_mats:
            matching_mats = all_mats

        # Deep parse top documents (capped to 2 to conserve AWS tokens)
        for m in matching_mats[:2]:
            if not m.get("raw_text_excerpt"):
                document_agent.analyze_document(m["id"])

        schedules = db.get_course_schedules(course_code=course_code)
        course_anns = db.get_announcements(term=term, course_code=course_code, limit=10)

        # Categorize documents
        question_sheets = []
        solution_sheets = []
        other_docs = []

        for m in matching_mats[:12]:
            questions = []
            if m.get("extracted_questions"):
                try:
                    questions = json.loads(m["extracted_questions"])
                except Exception:
                    pass

            doc_entry = {
                "id": m["id"],
                "title": m["title"],
                "course_code": m.get("course_code", course_code),
                "doc_type": m.get("doc_type", "UNKNOWN"),
                "is_solution": bool(m.get("is_solution", 0)),
                "topic": m.get("topic", ""),
                "summary": m.get("summary", ""),
                "key_questions": questions[:5],
                "downloaded": bool(m.get("downloaded")),
                "action_link": f"[OPEN_DOC:{m['id']}:{m['title']}]"
            }

            if doc_entry["is_solution"]:
                solution_sheets.append(doc_entry)
            elif "TUTORIAL" in doc_entry["doc_type"]:
                question_sheets.append(doc_entry)
            else:
                other_docs.append(doc_entry)

        finding_notes = []
        if matching_mats:
            test_titles = [m['title'] for m in matching_mats if any(k in m['title'].lower() for k in ['test', 'mock', 'exam'])]
            if test_titles:
                finding_notes.append(f"Test materials: {', '.join(test_titles[:3])}")
            syllabus_titles = [m['title'] for m in matching_mats if any(k in m['title'].lower() for k in ['hand00', 'syllabus', 'outline', 'schedule'])]
            if syllabus_titles:
                finding_notes.append(f"Syllabus file: {syllabus_titles[0]}")
        
        test_anns = [a['title'] for a in course_anns if any(k in a['title'].lower() for k in ['test', 'venue', 'quiz', 'exam', 'schedule'])]
        if test_anns:
            finding_notes.append(f"Notices: {', '.join(test_anns[:2])}")

        finding_str = f"Found {len(matching_mats)} documents for {course_code}. " + (" ".join(finding_notes) if finding_notes else f"Extracted {len(schedules)} semester milestones.")

        return {
            "subagent": "Document Specialist Agent",
            "status": "completed",
            "course_code": course_code,
            "task_type": task_type,
            "search_query": detail,
            "question_sheets_found": question_sheets,
            "solution_sheets_found": solution_sheets,
            "other_documents": other_docs[:10],
            "documents_found": matching_mats[:8],
            "schedules_extracted": schedules[:8],
            "relevant_notices": [{"title": a["title"], "date": a.get("posted_at", "")} for a in course_anns[:4]],
            "finding": finding_str
        }

    elif name == "task_planner_agent":
        hours = float(args.get("available_hours", 6.0))
        schedule = generate_day_schedule(term=term, available_hours=hours)
        tasks = db.get_tasks(term=term)
        return {
            "subagent": "Planner Engine",
            "status": "completed",
            "available_hours": hours,
            "scheduled_blocks": schedule[:6],
            "pending_tasks_count": len(tasks),
            "top_tasks": [t["title"] for t in tasks[:5]]
        }

    elif name == "create_action_task":
        import time
        title = args.get("title", "").strip()
        course_code = args.get("course_code") or "General"
        due_date = args.get("due_date") or ""
        est_min = int(args.get("estimated_minutes") or 60)
        priority = float(args.get("priority_score") or 9.0)
        notes = args.get("notes") or ""

        task_id = f"task_{int(time.time())}"
        db.upsert_task(
            task_id=task_id,
            title=title,
            source="assistant",
            course_code=course_code,
            due_date=due_date,
            estimated_minutes=est_min,
            priority_score=priority,
            notes=notes,
            term=term,
        )
        summary_text = f"Logged '{title}' (due: {due_date or 'today'}) with Priority {priority}."
        return {
            "subagent": "Task Engine",
            "status": "success",
            "task_id": task_id,
            "title": title,
            "due_date": due_date,
            "priority_score": priority,
            "finding": summary_text,
            "result_summary": summary_text
        }

    elif name == "task_ntulearn_specialist":
        raw_code = args.get("course_code") or ""
        norm_c = normalize_course_token(raw_code) if raw_code else None
        cm = re.search(r'\b([A-Z]{2,4}\d{4}[A-Z]?)\b', raw_code, re.IGNORECASE) if raw_code else None
        filter_code = norm_c if norm_c else (cm.group(1).upper() if cm else (raw_code or None))
        announcements = db.get_announcements(term=term, course_code=filter_code, limit=15)
        if query:
            q_lower = query.lower()
            announcements = [a for a in announcements if q_lower in (a.get("title") or "").lower() or q_lower in (a.get("body") or "").lower()]

        ann_results = []
        for a in announcements[:8]:
            ann_results.append({
                "id": a["id"],
                "title": a["title"],
                "course": a.get("course_code"),
                "posted": a.get("posted_at"),
                "body": (a.get("body") or "").strip()
            })

        return {
            "subagent": "NTULearn Specialist",
            "status": "completed",
            "course_code": raw_code,
            "announcements_found": len(ann_results),
            "announcements": ann_results,
            "finding": f"Checked notices for {raw_code or 'active semester'}: found {len(ann_results)} updates with full message content."
        }

    return {"error": f"Unknown tool {name}"}

def build_course_dossier_hub(courses: List[Dict[str, Any]], target_course: Optional[Dict[str, Any]], user_query: str, term: str = "26S1") -> str:
    """
    Constructs a Course-Sorted Offline Intelligence Dossier Hub.
    Groups course materials, assessment milestones, and announcements by module,
    with interactive tabs allowing the user to filter or view all modules.
    """
    import uuid
    hub_id = f"dossier-hub-{uuid.uuid4().hex[:6]}"
    target_clean = ""
    if target_course:
        tc_match = re.search(r'\b([A-Z]{2,4}\d{4}[A-Z]?)\b', target_course.get("course_code", ""), re.IGNORECASE)
        target_clean = tc_match.group(1).upper() if tc_match else ""

    q_lower = user_query.lower()
    is_test_query = any(w in q_lower for w in ["test", "exam", "quiz", "mock", "ca1", "ca2", "assessment"])
    is_sched_query = any(w in q_lower for w in ["schedule", "timetable", "dates", "calendar", "timeline", "week", "outline", "syllabus"])

    # Collect and sort unique courses
    unique_courses = []
    seen_codes = set()
    
    if target_course:
        t_code = target_clean or target_course.get("course_code", "")
        unique_courses.append(target_course)
        seen_codes.add(t_code)

    for c in courses:
        raw_c = c.get("course_code", "")
        cm = re.search(r'\b([A-Z]{2,4}\d{4}[A-Z]?)\b', raw_c, re.IGNORECASE)
        clean = cm.group(1).upper() if cm else raw_c
        if clean not in seen_codes:
            unique_courses.append(c)
            seen_codes.add(clean)

    # Build Navigation Tabs
    tab_pills = []
    default_tab = target_clean if target_clean else "all"
    
    active_all_cls = "active" if default_tab == "all" else ""
    tab_pills.append(f'''<button class="dossier-tab-btn {active_all_cls}" onclick="switchDossierCourse(\'{hub_id}\', \'all\')">&#128293; All Courses ({len(unique_courses)})</button>''')

    for c in unique_courses:
        raw_c = c.get("course_code", "")
        cm = re.search(r'\b([A-Z]{2,4}\d{4}[A-Z]?)\b', raw_c, re.IGNORECASE)
        clean = cm.group(1).upper() if cm else raw_c
        active_cls = "active" if default_tab == clean else ""
        tab_pills.append(f'''<button class="dossier-tab-btn {active_cls}" onclick="switchDossierCourse(\'{hub_id}\', \'{clean}\')">&#128218; {clean}</button>''')

    tabs_html = "\n".join(tab_pills)

    # Build Course Cards
    course_cards = []
    for c in unique_courses:
        raw_c = c.get("course_code", "")
        title = c.get("title", "")
        cm = re.search(r'\b([A-Z]{2,4}\d{4}[A-Z]?)\b', raw_c, re.IGNORECASE)
        clean = cm.group(1).upper() if cm else raw_c

        # Retrieve course data
        all_mats = db.get_all_materials(course_code=clean)
        tasks = [t for t in db.get_tasks(term=term) if clean in str(t.get("course_code", "")) or clean in str(t.get("title", ""))]
        anns = db.get_announcements(term=term, course_code=clean, limit=4)

        # Categorize materials
        syllabus_docs = [m for m in all_mats if any(k in m.get("title", "").lower() for k in ["hand00", "syllabus", "schedule", "outline", "overview"])]
        test_docs = [m for m in all_mats if any(k in m.get("title", "").lower() for k in ["test", "mock", "exam", "quiz", "ca1", "ca2"])]
        tut_docs = [m for m in all_mats if any(k in m.get("title", "").lower() for k in ["tutorial", "tut"])]
        lec_docs = [m for m in all_mats if any(k in m.get("title", "").lower() for k in ["hand", "lec", "slide", "lecture"]) and m not in syllabus_docs]

        # Top highlight doc
        primary_doc = syllabus_docs[0] if syllabus_docs else (test_docs[0] if test_docs else (all_mats[0] if all_mats else None))

        # Build Materials List HTML
        doc_links = []
        if syllabus_docs:
            for s in syllabus_docs[:2]:
                doc_links.append(f'''<a href="/api/materials/file/{s["id"]}" target="_blank" class="dossier-doc-link"><span class="dossier-tag tag-syllabus">Syllabus</span> <strong>{s["title"]}</strong></a>''')
        if test_docs:
            for td in test_docs[:2]:
                doc_links.append(f'''<a href="/api/materials/file/{td["id"]}" target="_blank" class="dossier-doc-link"><span class="dossier-tag tag-test">Mock Exam</span> <strong>{td["title"]}</strong></a>''')
        if tut_docs:
            for tu in tut_docs[:2]:
                doc_links.append(f'''<a href="/api/materials/file/{tu["id"]}" target="_blank" class="dossier-doc-link"><span class="dossier-tag tag-tutorial">Problem Set</span> <strong>{tu["title"]}</strong></a>''')
        elif lec_docs:
            for ld in lec_docs[:2]:
                doc_links.append(f'''<a href="/api/materials/file/{ld["id"]}" target="_blank" class="dossier-doc-link"><span class="dossier-tag tag-lecture">Lecture</span> <strong>{ld["title"]}</strong></a>''')

        if not doc_links:
            doc_links.append('<span style="color:var(--text-muted); font-size:0.75rem;">No direct PDF files indexed for this module yet.</span>')

        docs_col_html = "".join(doc_links)

        # Build Tasks & Assessments HTML
        task_items = []
        if tasks:
            for t in tasks[:3]:
                due = t.get("due_date") or "Upcoming"
                score = t.get("priority_score", 7.0)
                task_items.append(f'''<div class="dossier-task-item"><span class="dossier-score-tag">Score: {score}</span> <div><strong>{t["title"]}</strong> <div style="font-size:0.7rem; color:var(--text-muted);">Due: {due}</div></div></div>''')
        else:
            task_items.append(f'''<div class="dossier-task-item"><span class="dossier-score-tag" style="background:rgba(0,242,254,0.15); color:var(--accent-cyan);">Info</span> <div>Continuous Assessment / Midterms during AY2026/27 Sem 1</div></div>''')

        tasks_col_html = "".join(task_items)

        # Build Announcements HTML
        ann_items = []
        if anns:
            for a in anns[:2]:
                posted = a.get("posted_at") or "Recent"
                ann_items.append(f'''<div class="dossier-ann-item"><strong>{a["title"]}</strong><span style="font-size:0.68rem; color:var(--text-muted); display:block; margin-top:2px;">{posted}</span></div>''')
        else:
            ann_items.append('<div style="color:var(--text-muted); font-size:0.75rem;">No urgent announcements posted.</div>')

        anns_col_html = "".join(ann_items)

        is_card_visible = "style=\"display:flex;\"" if (default_tab == "all" or default_tab == clean) else "style=\"display:none;\""

        card_html = f'''
        <div class="dossier-course-card" data-course="{clean}" {is_card_visible}>
          <div class="dossier-card-head">
            <div>
              <span class="dossier-course-badge">{clean}</span>
              <span class="dossier-course-title">{title}</span>
            </div>
            <span class="dossier-mats-count">{len(all_mats)} files indexed</span>
          </div>

          <div class="dossier-card-grid">
            <!-- Col 1: Materials -->
            <div class="dossier-col">
              <div class="dossier-col-title">&#128196; Essential Course Files</div>
              <div class="dossier-col-body">{docs_col_html}</div>
            </div>

            <!-- Col 2: Assessments -->
            <div class="dossier-col">
              <div class="dossier-col-title">&#9200; Assessments & Tasks</div>
              <div class="dossier-col-body">{tasks_col_html}</div>
            </div>

            <!-- Col 3: Announcements -->
            <div class="dossier-col">
              <div class="dossier-col-title">&#128227; Official Notices</div>
              <div class="dossier-col-body">{anns_col_html}</div>
            </div>
          </div>
        </div>
        '''
        course_cards.append(card_html)

    cards_html = "".join(course_cards)

    hub_title = f"Course Intelligence Hub: {target_clean}" if target_clean else "Course Intelligence Hub (All Enrolled Modules)"
    badge_label = f"{target_clean} Module" if target_clean else "All Modules"

    return f'''<!-- COURSE_DOSSIER_HUB -->
<div class="course-dossier-hub" id="{hub_id}">
  <div class="dossier-header-bar">
    <div>
      <h3 style="margin:0; font-size:1.05rem; color:#fff; display:flex; align-items:center; gap:8px;">
        &#9889; {hub_title}
      </h3>
      <div style="font-size:0.74rem; color:var(--text-muted); margin-top:2px;">
        Sorted by Course Module &bull; Direct SQLite Database Query &bull; Zero Cloud Token Cost
      </div>
    </div>
    <span class="dossier-mode-badge">&#9889; {badge_label}</span>
  </div>

  <div class="dossier-tabs-strip">
    {tabs_html}
  </div>

  <div class="dossier-cards-list">
    {cards_html}
  </div>
</div>
'''


def clean_voice_summary_text(text: str) -> str:
    """Clean text for text-to-speech by stripping HTML, markdown, URLs, and all emojis."""
    if not text:
        return ""
    clean = re.sub(r'<[^>]+>', ' ', text)
    clean = re.sub(r'\[OPEN_DOC:[^\]]+\]', ' ', clean)
    clean = re.sub(r'\[VOICE_SUMMARY:\s*', ' ', clean, flags=re.IGNORECASE)
    clean = re.sub(r'\]', ' ', clean)
    clean = re.sub(r'#+\s*', '', clean)
    clean = re.sub(r'[*_`~|]', '', clean)
    clean = re.sub(r'^\s*[-•*]\s+', '', clean, flags=re.MULTILINE)
    clean = re.sub(r'https?:\/\/\S+', '', clean)
    emoji_pattern = re.compile(
        r'[\U0001F1E0-\U0001F1FF\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF'
        r'\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF'
        r'\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\u2600-\u26FF\u2700-\u27BF]+',
        flags=re.UNICODE
    )
    clean = emoji_pattern.sub('', clean)
    clean = re.sub(r'&#\d+;', '', clean)
    clean = re.sub(r'&[a-z]+;', '', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean

def extract_and_strip_voice_summary(reply_text: str) -> Tuple[str, str]:
    """
    Extracts the model-generated [VOICE_SUMMARY: ...] tag for text-to-speech,
    and cleanly strips it from the visual markdown response.
    """
    if not reply_text:
        return "", ""
    
    voice_summary = ""
    pattern = re.compile(r'\[VOICE_SUMMARY:\s*(.*?)\]', flags=re.DOTALL | re.IGNORECASE)
    match = pattern.search(reply_text)
    if match:
        voice_summary = clean_voice_summary_text(match.group(1))
        cleaned_reply = pattern.sub('', reply_text).strip()
    else:
        cleaned_reply = reply_text.strip()
    
    # If the model did not generate [VOICE_SUMMARY: ...], extract a clean 3-4 sentence prose fallback
    if not voice_summary or len(voice_summary) < 6:
        base = re.sub(r'<div class="subagent-[^"]*">.*?</div>', '', cleaned_reply, flags=re.DOTALL)
        base = re.sub(r'#+ [^\r\n]+', ' ', base)
        base = re.sub(r'^\s*\d+[\.\)] [^\r\n]+', ' ', base, flags=re.MULTILINE)
        base = re.sub(r'[-–—]{2,}', ' ', base)
        clean = clean_voice_summary_text(base)
        sentences = re.split(r'(?<=[.!?])\s+', clean)
        valid = [s.strip() for s in sentences if len(s.strip()) > 10 and not s.strip().startswith('---') and ':' not in s.strip()[:20]]
        voice_summary = ' '.join(valid[:4]).strip()
        if voice_summary and voice_summary[-1] not in '.!?':
            voice_summary += '.'
        if not voice_summary:
            voice_summary = clean.strip()
            
    return cleaned_reply, voice_summary

def execute_chat_query(user_query: str, term: str = "26S1", session_id: str = "default") -> Dict[str, Any]:
    courses = db.get_all_courses(term=term)
    courses_str = ", ".join([f"{c['course_code']} ({c.get('title') or c['course_code']})" for c in courses])

    # ── Multi-turn history retrieval ──────────────────────────────────────────
    history = db.get_chat_history(session_id=session_id, limit=8)
    messages: List[Dict[str, Any]] = []

    for h in history:
        r = h.get("role", "user")
        txt = h.get("content", "").strip()
        txt = re.sub(r'<div class="subagent-[^"]*">.*?</div>', '', txt, flags=re.DOTALL)
        txt = re.sub(r'<[^>]+>', '', txt).strip()
        if not txt:
            continue

        if messages and messages[-1]["role"] == r:
            messages[-1]["content"][0]["text"] += f"\n\n{txt}"
        else:
            messages.append({"role": r, "content": [{"text": txt}]})

    while messages and messages[0]["role"] != "user":
        messages.pop(0)

    # Resolve course target from user prompt and multi-turn context
    matched_course = extract_course_from_query(user_query, history, courses)
    if matched_course:
        target_course_code = matched_course["course_code"]
        target_course_title = matched_course.get("title") or target_course_code
        clean_target = re.search(r'\b([A-Z]{2,4}\d{4}[A-Z]?)\b', target_course_code)
        target_code = clean_target.group(1) if clean_target else target_course_code
        target_line = f"Identified Course Target: {target_course_code} ({target_code} - {target_course_title})"
    else:
        target_course_code = None
        target_course_title = "All Enrolled Modules"
        target_code = None
        target_line = "Identified Course Target: None (General academic question across all enrolled modules)"

    # Active Timetable Context
    active_tt = getattr(db, "get_active_timetable", lambda term: None)(term=term) if hasattr(db, "get_active_timetable") else None
    tt_summary = ""
    if active_tt and active_tt.get("parsed_data"):
        pdata = active_tt["parsed_data"]
        sname = pdata.get("student_name", "Student")
        slots_list = pdata.get("weekly_slots", [])
        slots_text = "; ".join([f"{s['day']} {s['start_time']}-{s['end_time']} {s['course_code']} {s['event_type']} @ {s.get('venue','TBA')}" for s in slots_list])
        exams_list = [f"{c['course_code']}: {c['exam_schedule']}" for c in pdata.get("courses", []) if "not applicable" not in c.get("exam_schedule", "").lower()]
        exams_text = "; ".join(exams_list)
        tt_summary = f"\nStudent Profile: {sname}\nOfficial Timetable Weekly Slots: {slots_text}\nOfficial Final Exam Dates: {exams_text}\n"

    # Current context grounding
    now_dt = datetime.now()
    day_name = now_dt.strftime('%A')
    date_str = now_dt.strftime('%d %B %Y')
    current_context = f"""Current Date & Time: {day_name}, {date_str} (Today is strictly {day_name}, AY2026/27 Semester 1 - Week 4)
Active Term: {term}
{target_line}
Enrolled Modules: {courses_str}
{tt_summary}
Student Question: {user_query}
CRITICAL INSTRUCTION: Today is {day_name}, {date_str} (07 September 2026 is strictly {day_name}). Answer the student's question accurately based on the identified context. If a specific course is asked about, focus on that course. If the question is general or cross-module (e.g. study plan, urgent deadlines across all courses, daily timetable, greeting), address all relevant modules across their academic workload rather than limiting your answer to one course. At the very end of your response, you MUST append [VOICE_SUMMARY: 3-4 complete spoken sentences answering the question naturally with zero emojis and zero markdown]."""

    if messages and messages[-1]["role"] == "user":
        if target_code:
            messages[-1]["content"][0]["text"] += f"\n\nTarget Module: {target_code}\nFollow-up Question: {user_query}"
        else:
            messages[-1]["content"][0]["text"] += f"\n\nFollow-up Question: {user_query}"
    else:
        messages.append({"role": "user", "content": [{"text": current_context}]})

    delegation_steps = []
    final_reply = ""

    # 1. PRIORITY 1: AWS Bedrock Claude 3.5 Sonnet
    provider_name_used = "Offline Hybrid Engine"
    if bedrock_client.is_ready():
        try:
            tool_config = {"tools": TOOL_DEFINITIONS}
            turn = 0
            while turn < 3:
                turn += 1
                resp = bedrock_client._client.converse(
                    modelId=LEAD_MODEL_ID,
                    messages=messages,
                    system=[{"text": LEAD_SYSTEM_PROMPT}],
                    toolConfig=tool_config,
                    inferenceConfig={"maxTokens": 2048, "temperature": 0.2}
                )
                output_msg = resp.get("output", {}).get("message", {})
                messages.append(output_msg)
                stop_reason = resp.get("stopReason")
                content_blocks = output_msg.get("content", [])

                if stop_reason == "tool_use":
                    tool_result_contents = []
                    for block in content_blocks:
                        if "toolUse" in block:
                            t_use = block["toolUse"]
                            subagent_out = execute_subagent_tool(t_use["name"], t_use.get("input", {}), term=term)
                            delegation_steps.append({
                                "agent": subagent_out.get("subagent", t_use["name"]),
                                "action": f"Delegated to {t_use['name']}",
                                "input": t_use.get("input", {}),
                                "result_summary": subagent_out.get("finding") or str(subagent_out)[:140]
                            })
                            tool_result_contents.append({
                                "toolResult": {
                                    "toolUseId": t_use["toolUseId"],
                                    "content": [{"json": subagent_out}],
                                    "status": "success"
                                }
                            })
                    messages.append({"role": "user", "content": tool_result_contents})
                else:
                    for block in content_blocks:
                        if "text" in block:
                            final_reply += block["text"]
                    provider_name_used = "AWS Bedrock (Claude 3.5)"
                    
                    usage = resp.get("usage", {})
                    pt = usage.get("inputTokens", 0)
                    ct = usage.get("outputTokens", 0)
                    tt = usage.get("totalTokens", pt + ct)
                    lat = resp.get("metrics", {}).get("latencyMs", 1000)
                    try:
                        db.record_token_usage("AWS Bedrock", LEAD_MODEL_ID, pt, ct, tt, lat, "success", "chat")
                    except Exception:
                        pass
                    break
        except Exception as e:
            print(f"[Agents] Bedrock priority notice ({e}). Auto-falling back to Groq...")
            final_reply = ""

    # 2. PRIORITY 2 & 3: Groq Cloud with Google Gemini Fallback
    if not final_reply and llm_provider.is_ready():
        try:
            openai_msgs = []
            for h in history[-4:]:
                r = h.get("role", "user")
                t = re.sub(r'<div class="subagent-[^"]*">.*?</div>', '', h.get("content", ""), flags=re.DOTALL)
                t = re.sub(r'<[^>]+>', '', t).strip()
                if t:
                    openai_msgs.append({"role": r, "content": t})
            openai_msgs.append({"role": "user", "content": current_context})

            max_turns = 3
            turn = 0
            while turn < max_turns:
                turn += 1
                tools_to_use = None if (delegation_steps and turn >= 2) else OPENAI_TOOL_DEFINITIONS
                res_msg, provider = llm_provider.chat_completion(
                    messages=openai_msgs,
                    tools=tools_to_use,
                    system_prompt=LEAD_SYSTEM_PROMPT,
                    temperature=0.2,
                    max_tokens=2048
                )
                provider_name_used = provider

                if not res_msg:
                    break

                tool_calls = getattr(res_msg, "tool_calls", None)
                if tool_calls and len(tool_calls) > 0:
                    openai_msgs.append({
                        "role": "assistant",
                        "content": res_msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            } for tc in tool_calls
                        ]
                    })

                    for tc in tool_calls:
                        t_name = tc.function.name
                        try:
                            t_input = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                        except Exception:
                            t_input = {}

                        subagent_out = execute_subagent_tool(t_name, t_input, term=term)
                        delegation_steps.append({
                            "agent": subagent_out.get("subagent", t_name),
                            "action": f"Delegated to {t_name}",
                            "input": t_input,
                            "result_summary": subagent_out.get("finding") or str(subagent_out)[:140]
                        })

                        openai_msgs.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(subagent_out)
                        })
                else:
                    if res_msg.content:
                        final_reply = res_msg.content
                    break

        except Exception as e:
            print(f"[Agents] Error during Groq/Gemini execution ({e}), falling back to local specialist...")
            final_reply = ""

    # If offline, expired credentials, or fallback required
    if not final_reply or "Bedrock tool execution error" in final_reply:
        q_clean = user_query.strip().lower()
        all_courses = db.get_all_courses(term=term)
        provider_name_used = "Lead Orchestrator (Local Engine)"

        # Disambiguate user intent
        is_greeting = bool(re.search(r'^(hello|hi|hey|good\s*(morning|afternoon|evening)|who\s*are\s*you|howdy|sup)\b', q_clean))
        is_study_plan = any(w in q_clean for w in ["study plan", "plan for tonight", "plan my day", "3-hour", "3 hour", "how should i study", "schedule my night"])
        is_urgent = any(w in q_clean for w in ["urgent deadline", "deadlines", "urgent", "recent announcement", "check announcement", "important notice"])
        is_tutorial_query = any(w in q_clean for w in ["what tutorial", "which tutorial", "tutorial do i need", "what do i need to do right now"])

        if is_greeting:
            final_reply = """Hello! I am your **CogniFlow Multi-Agent Lead Orchestrator**.

I am actively tracking your **AY2026/27 Semester 1 (26S1)** academic schedule across all your enrolled modules:
- 📊 **MH2500**: Probability & Introduction to Statistics
- 💻 **SC2207**: Introduction to Databases
- ⚡ **SC2001**: Algorithm Design and Analysis
- 🌿 **CC0006**: Sustainability: Society, Economy & Environment
- 📐 **MH2802**: Linear Algebra for Scientists
- 💼 **ML0004**: Career & Entrepreneurial Skills

**What would you like to work on?**
- 📄 Ask about any module: *"What is tested for SC2207?"*, *"CC0006 project topics"*, or *"MH2500 mock exams"*
- ⏰ Plan your time: *"Create an optimized 3-hour study plan for tonight"*
- 📅 Check today's classes: *"What tutorial do I need today?"* or *"Check urgent deadlines"*
"""
            speech_sum = "Hello! I am your CogniFlow Lead Orchestrator. I am actively tracking your enrolled courses, lectures, and deadlines for Semester 1. What would you like to work on?"

        elif is_study_plan:
            sched = generate_day_schedule(term=term, available_hours=3.0)
            plan_blocks = []
            if sched:
                for idx, b in enumerate(sched[:4], 1):
                    plan_blocks.append(f"{idx}. **{b['start']} – {b['end']} ({b['duration_mins']}m)**: **{b['title']}** ({b.get('course_code', 'General')})")
            else:
                plan_blocks = [
                    "1. **08:00 PM – 09:00 PM (60m)**: **SC2207 Lab 1 Preparation** (Database Schema & Queries)",
                    "2. **09:15 PM – 10:15 PM (60m)**: **MH2500 Probability Problem Set** (Mock Exam Review)",
                    "3. **10:30 PM – 11:00 PM (30m)**: **CC0006 Project Topic Review** (Discussion & Reading)"
                ]

            plan_text = "\n".join(plan_blocks)
            final_reply = f"""### ⚡ Optimized 3-Hour Study Plan for Tonight

Based on your upcoming deadlines and enrolled course priorities, here is your recommended schedule:

{plan_text}

💡 **Action Tip**: Keep breaks to 15 minutes between deep-work sessions. All blocks have been prioritized based on deadline urgency!
"""
            speech_sum = "Here is an optimized 3-hour study plan for tonight, prioritizing your highest-urgency tasks and mock exam preparation with short recovery breaks."

        elif is_urgent and not matched_course:
            tasks = db.get_tasks(term=term)
            urgent_tasks = sorted([t for t in tasks if t.get("status") != "completed"], key=lambda x: x.get("priority_score", 0), reverse=True)[:5]
            recent_anns = db.get_announcements(term=term, limit=4)

            task_lines = []
            for t in urgent_tasks:
                task_lines.append(f"- **[{t.get('course_code', 'General')}] {t['title']}** &bull; Due: `{t.get('due_date') or 'Upcoming'}` (Score: **{t.get('priority_score', 7.0)}**)")
            
            ann_lines = []
            for a in recent_anns:
                ann_lines.append(f"- **{a.get('course_code', '')}**: *{a['title']}* ({a.get('posted_at') or 'Recent'})")

            final_reply = f"""### 🚨 Urgent Deadlines & Priority Action Items

Here is your cross-module urgency digest for **AY2026/27 Semester 1**:

**Top Priority Deadlines**:
{chr(10).join(task_lines) if task_lines else "- No critical pending deadlines logged."}

**Recent Official Notices**:
{chr(10).join(ann_lines) if ann_lines else "- No urgent announcements posted."}
"""
            speech_sum = "Here are your urgent deadlines and recent announcements across your enrolled modules, ordered by priority score."

        elif is_tutorial_query and not matched_course:
            final_reply = """### 📅 Today's Schedule & Tutorial Guide

According to your official timetable for **Monday, 07 September 2026**:
- **09:30 AM – 10:30 AM**: **MH2500** Probability Lecture @ `LT1A`
- **11:30 AM – 12:30 PM**: **SC2207** Databases Lecture @ `LT2A`
- **02:30 PM – 04:30 PM**: **CC0006** Sustainability Tutorial @ `TR+15`

**Required Tutorial Materials Today**:
- 📄 **CC0006**: *Tutorial Group Student Name Registration & Week 4 Discussion Guide*
- 📄 **SC2207**: *Lab 1 Preparation & Guidelines*

**Upcoming Tutorials Later This Week**:
- **SC2207 Tutorial**: Tuesday 10:30 AM
- **SC2001 Lab**: Wednesday 02:30 PM
- **ML0004 Tutorial**: Thursday 09:30 AM
"""
            speech_sum = "Today you have an MH2500 lecture at 9:30 AM, SC2207 lecture at 11:30 AM, and your CC0006 tutorial this afternoon from 2:30 PM at TR+15."

        else:
            final_reply = build_course_dossier_hub(all_courses, matched_course, user_query, term=term)
            provider_name_used = "Local Course Intelligence Hub"
            if matched_course:
                cm = re.search(r'\b([A-Z]{2,4}\d{4}[A-Z]?)\b', matched_course.get("course_code", ""))
                code_tag = cm.group(1).upper() if cm else matched_course.get("course_code", "")
                speech_sum = f"Here is the course overview, essential files, and indexed notices for {code_tag}."
            else:
                speech_sum = "Here is the course intelligence overview across all your enrolled modules."

        # Save to database
        db.save_chat_message("user", user_query, agent_name="User", session_id=session_id)
        db.save_chat_message("assistant", final_reply, agent_name="Lead Orchestrator (Local Hub)", session_id=session_id)

        return {
            "reply": final_reply,
            "response": final_reply,
            "speech_summary": speech_sum,
            "agent": "Lead Orchestrator (Local Hub)",
            "model": provider_name_used,
            "term": term,
            "delegation_steps": delegation_steps,
            "subagents_called": [s["agent"] for s in delegation_steps],
            "execution_mode": "offline_course_hub",
        }

    # Build visible compact Sub-Agent Delegation badges
    delegation_callout = ""
    if delegation_steps:
        sub_badges = []
        for s in delegation_steps:
            agent_title = s.get("agent", "Specialist Agent")
            summary = s.get("result_summary", "")
            if len(summary) > 130:
                summary = summary[:127] + "..."
            sub_badges.append(
                f'<div class="subagent-compact-badge">'
                f'<span class="subagent-pulse"></span>'
                f'<span class="subagent-badge-title">⚡ {agent_title}:</span> '
                f'<span class="subagent-badge-desc">{summary}</span>'
                f'</div>'
            )
        delegation_callout = "".join(sub_badges) + "\n\n"

    # Failsafe check: if user asked to add a task/interview but tool wasn't triggered
    task_added_by_tool = any(s.get("agent") in ["create_action_task", "Task Engine"] for s in delegation_steps)
    if not task_added_by_tool and re.search(r'\b(add|create|schedule|log|set|have)\s+(a\s+)?(task|interview|reminder|meeting|appointment)\b|\binterview\b', user_query, re.IGNORECASE):
        import time
        if "interview" in user_query.lower():
            t_match = re.search(r'(\d{1,2}(?::\d{2})?\s*(?:am|pm)?|\d{4}\s*hrs?)', user_query, re.IGNORECASE)
            time_str = f" ({t_match.group(1).upper()})" if t_match else ""
            fallback_title = f"Interview{time_str}"
        else:
            m_title = re.search(r'\b(?:task\s+for|task\s+that|task\s+to|task:?)\s+([^\.,;\n\?]+)', user_query, re.IGNORECASE)
            fallback_title = m_title.group(1).strip().capitalize() if m_title else "Scheduled Action Item"
            if len(fallback_title) > 50:
                fallback_title = fallback_title[:47] + "..."

        due_str = f"{datetime.now().strftime('%Y-%m-%d')} 16:00" if ("4:00" in user_query or "4pm" in user_query.lower() or "4 pm" in user_query.lower()) else f"{datetime.now().strftime('%Y-%m-%d')}"
        
        task_id = f"task_{int(time.time())}"
        db.upsert_task(
            task_id=task_id,
            title=fallback_title,
            source="assistant",
            course_code="General",
            due_date=due_str,
            estimated_minutes=60,
            priority_score=9.5,
            notes="Logged via Lead Orchestrator",
            term=term,
        )
        delegation_steps.append({
            "agent": "Task Engine",
            "result_summary": f"Logged '{fallback_title}' into Prioritized Action Items."
        })
        # Rebuild delegation badges
        sub_badges = []
        for s in delegation_steps:
            agent_title = s.get("agent", "Specialist Agent")
            summary = s.get("result_summary", "")
            if len(summary) > 130:
                summary = summary[:127] + "..."
            sub_badges.append(
                f'<div class="subagent-compact-badge">'
                f'<span class="subagent-pulse"></span>'
                f'<span class="subagent-badge-title">⚡ {agent_title}:</span> '
                f'<span class="subagent-badge-desc">{summary}</span>'
                f'</div>'
            )
        delegation_callout = "".join(sub_badges) + "\n\n"

    composed_reply = f"{delegation_callout}{final_reply}"

    final_clean_reply, speech_sum = extract_and_strip_voice_summary(composed_reply)

    db.save_chat_message("user", user_query, agent_name="User", session_id=session_id)
    db.save_chat_message("assistant", final_clean_reply, agent_name="Lead Orchestrator", session_id=session_id)

    return {
        "reply": final_clean_reply,
        "speech_summary": speech_sum,
        "agent": "Lead Orchestrator",
        "delegation_steps": delegation_steps,
        "model": provider_name_used,
        "term": term,
    }
