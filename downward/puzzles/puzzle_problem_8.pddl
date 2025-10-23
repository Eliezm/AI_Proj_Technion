;; ==============================
;; puzzle8_problem.pddl
;; ==============================
(define (problem puzzle8-problem)
  (:domain puzzle)

  (:objects
    ;; eight numbered tiles
    t1 t2 t3 t4 t5 t6 t7 t8 - tile
    ;; nine board positions (3×3 grid)
    p1 p2 p3 p4 p5 p6 p7 p8 p9 - position
  )

  ;; ┌───┬───┬───┐
  ;; │ p1│ p2│ p3│
  ;; ├───┼───┼───┤
  ;; │ p4│ p5│ p6│
  ;; ├───┼───┼───┤
  ;; │ p7│ p8│ p9│
  ;; └───┴───┴───┘

  (:init
    ;; (Example of a “scrambled” start; replace as desired)
    ;; You must put exactly eight (at t• p•) facts and one (empty p•).
    ;; In a true 8-puzzle, exactly one position is empty at a time.
    (at t1 p1)
    (at t2 p2)
    (at t3 p3)
    (at t4 p4)
    (at t5 p5)
    (at t6 p6)
    (at t7 p7)
    (at t8 p8)
    (empty p9)
  )

  (:goal
    ;; (Example “goal” configuration: tiles 1–8 in order, blank at p9)
    (and
      (at t1 p1)
      (at t2 p2)
      (at t3 p3)
      (at t4 p4)
      (at t5 p5)
      (at t6 p6)
      (at t7 p7)
      (at t8 p8)
      ;; Note: we do *not* need “(empty p9)” here, because
      ;; if all eight tiles occupy p1–p8, exactly p9 is empty by necessity.
    )
  )
)