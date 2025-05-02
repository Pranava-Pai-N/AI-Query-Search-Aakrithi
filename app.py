from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer, util
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

model = SentenceTransformer("all-MiniLM-L6-v2")

class Routine(BaseModel):
    time: str
    content: str

class Post(BaseModel):
    postId: str
    title: str
    description: str
    routines: Optional[List[Routine]] = []
    filters: List[str]
    type: str

class SearchRequest(BaseModel):
    prompt: str
    posts: List[Post]

@app.get("/")
async def root():
    return {"message": "Welcome to the Post Search API"}


@app.post("/search")
def search_posts(data: SearchRequest):
    prompt = data.prompt
    posts = data.posts

    if not posts:
        return {"message": "No posts provided", "results": []}

    def flatten(post: Post):
        routine_text = " ".join([f"{r.time} {r.content}" for r in post.routines]) if post.routines else ""
        filters_text = " ".join(post.filters)
        return f"{post.title} {post.description} {routine_text} {filters_text} {post.type}"

    flattened_posts = [flatten(post) for post in posts]
    post_embeddings = model.encode(flattened_posts, convert_to_tensor=True)
    prompt_embedding = model.encode(prompt, convert_to_tensor=True)

    similarities = util.cos_sim(prompt_embedding, post_embeddings)[0]
    similarity_scores = similarities.tolist()

    top_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)
    threshold = 0.5
    top_results = []

    for idx in top_indices[:3]:
        score = similarity_scores[idx]
        if score >= threshold:
            top_results.append({
                "postId": posts[idx].postId,
                "type": posts[idx].type
            })

    if top_results:
        return {"results": top_results}
    else:
        return {"results": []}


