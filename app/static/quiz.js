/* Interactive multiple-choice quiz: pick an option, reveal the answer and
   explanation, then track a running score. Uses event delegation so it keeps
   working when Datastar swaps the quiz card in place. */
(() => {
  function answer(question, selected) {
    if (question.dataset.answered === "true") return;

    const options = question.querySelectorAll("[data-quiz-option]");
    const selectedCorrect = selected.dataset.correct === "true";

    options.forEach((option) => {
      option.disabled = true;
      option.classList.remove("quiz-option-selected");
    });
    selected.classList.add("quiz-option-selected");
    options.forEach((option) => {
      if (option.dataset.correct === "true") {
        option.classList.add("quiz-option-correct");
      }
    });
    if (!selectedCorrect) selected.classList.add("quiz-option-wrong");

    const explanation = question.querySelector("[data-explanation]");
    if (explanation) explanation.hidden = false;

    question.dataset.answered = "true";
    updateSummary();
  }

  function updateSummary() {
    const quiz = document.querySelector("[data-quiz]");
    if (!quiz) return;

    const questions = quiz.querySelectorAll("[data-question]");
    const answered = quiz.querySelectorAll('[data-question][data-answered="true"]');
    const summary = quiz.querySelector("[data-quiz-summary]");
    const score = quiz.querySelector(".quiz-score");
    if (!summary || !score || answered.length === 0) return;

    let correct = 0;
    answered.forEach((question) => {
      const selected = question.querySelector(".quiz-option-selected");
      if (selected && selected.dataset.correct === "true") correct++;
    });

    const total = questions.length;
    if (answered.length === total) {
      score.textContent =
        correct === total
          ? `Perfect! You answered all ${total} questions correctly.`
          : `You got ${correct} of ${total} questions right.`;
    } else {
      score.textContent = `Score so far: ${correct}/${answered.length}`;
    }
    summary.hidden = false;
  }

  function resetQuiz() {
    const quiz = document.querySelector("[data-quiz]");
    if (!quiz) return;

    quiz.querySelectorAll("[data-question]").forEach((question) => {
      question.dataset.answered = "false";
      question.querySelectorAll("[data-quiz-option]").forEach((option) => {
        option.disabled = false;
        option.classList.remove(
          "quiz-option-selected",
          "quiz-option-correct",
          "quiz-option-wrong"
        );
      });
      const explanation = question.querySelector("[data-explanation]");
      if (explanation) explanation.hidden = true;
    });

    const summary = quiz.querySelector("[data-quiz-summary]");
    if (summary) summary.hidden = true;
  }

  document.addEventListener("click", (event) => {
    if (event.target.closest("[data-quiz-reset]")) {
      resetQuiz();
      return;
    }
    const option = event.target.closest("[data-quiz-option]");
    if (!option) return;
    const question = option.closest("[data-question]");
    if (!question) return;
    answer(question, option);
  });
})();
