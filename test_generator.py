from app.generator import TextGenerator

generator = TextGenerator()

result = generator.generate(
    prompt="ROMEO:",
    max_new_chars=200,
    temperature=0.8,
    top_k=10
)

print("\nGenerated Text:\n")
print(result)
