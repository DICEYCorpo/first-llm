import json
from fastapi import FastAPI, Query
from openai import OpenAI
from dotenv import load_dotenv
import uvicorn

load_dotenv()

app = FastAPI()
client = OpenAI()

@app.get("/generatereport")
async def generate_report(
    gpa: float = Query(..., description="Student's GPA"),
    location: str = Query(..., description="Preferred country for study"),
    course_level: str = Query(..., description="Desired level of study, e.g., Bachelor’s, Master’s"),
    fee_range: str = Query(..., description="The 'Fee Range' a student can afford for a potential college"),
    course_preference: str = Query(..., description="General field of study the student is interested in"),
    course1: str = Query(..., description="First course preference"),
    course2: str = Query(..., description="Second course preference"),
    course3: str = Query(..., description="Third course preference"),
    regional_choice: str = Query(..., description="Preferred region within the US, e.g., Northeast, West Coast")
):
    formatted_text = f"""1) GPA: {gpa}
2) Location: {location}
3) Course Level: {course_level}
4) Fee Range: {fee_range}
5) Course Preference: {course_preference}
6) Best 3 course preference under that subject: {course1}, {course2}, {course3}
7) Regional Choice inside US: {regional_choice}"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {
                "role": "system",
                "content": "Admission Councilor\n\nSuppose you are a very friendly admission councilor.\nPlease provide a detailed analysis of a student profile for college admission purposes. Your report should include strengths, areas for improvement, interests, and opportunities based on the input provided.  While writing the report make sure you are addressing the student in a friendly behavior.  Make sure when the student is reading it, he or she should feel like it is written for them. No need to provide prefix paragraph and conclusion paragraph\n\nStudent Profile Inputs:\n\nGPA: [Student's GPA]\nLocation Preference: [Preferred country for study]\nCourse Level: [Desired level of study, e.g., Bachelor’s, Master’s]\nFee Range: [The 'Fee Range' a student can afford for a potential college]\nCourse Preference: [General field of study the student is interested in]\nTop 3 Course Preferences: [List three specific areas of interest within the broader course preference]\nRegional Preference in the US: [Preferred region within the US, e.g., Northeast, West Coast]\n\nWriting Structure: (STRICTLY FOLLOW THIS FORMAT)\n\nStrengths: \n[Identify and explain the strengths in the student's profile, such as academic performance, clarity of educational goals, and financial planning.]\n\nAreas for Improvement: \n[Suggest areas where the student could enhance their profile, including geographical focus, exploration of specific academic interests, and inclusion of additional standard academic metrics if applicable. You do not want student to feel they have to do more work. Hence this output needs to be bit re-tweaked so that there's a more 'Assertive' output]\n\nInterests: \n[Discuss how the student’s interests align with their chosen field of study and suggest related academic subjects they might enjoy.]\n\nOpportunities: \n[Highlight potential opportunities for the student based on their profile, such as suitable colleges, scholarships, internships, and extracurricular activities that match their academic and career goals. \"opportunities\" are meant for future employment based opportunities and NOT which schools could be a good match.   We are NOT to give any school matchings .. as this is just a holistic output without school pairings during this output.  For school recommendations, we got this covered internally and not expecting you to telling us about the schools]\n\n\nNO NEED TO WRITE ANYTHING EXTRA OUTSIDE OF THESE POINTS AFTER FINISH WRITING ABOVE POINTS.\n\nObjective:\n\nProvide a holistic report that aids the student in refining their college search and application strategy, ensuring they are well-prepared to pursue their educational objectives effectively.\n\n\nReturn it in this JSON format:\n\n{{\n  \"Strengths\": [ Sentence 1, Sentence 2 .....],\n  \"Areas_for_Improvement\":  [ Sentence 1, Sentence 2 .....],\n  \"Interests\":  [ Sentence 1, Sentence 2 .....],\n  \"Opportunities\": [ Sentence 1, Sentence 2 .....]\n}}"
            },
            {
                "role": "user",
                "content": formatted_text
            }
        ],
        temperature=1,
        max_tokens=8000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    respone = response.choices[0].message.content
    proper_json = json.loads(respone)
    for key, value in proper_json.items():
        if isinstance(value, list):
            proper_json[key] = [item.replace("\n", "") for item in value]

    return proper_json

