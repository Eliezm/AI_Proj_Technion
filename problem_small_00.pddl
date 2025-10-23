(define (problem blocks-world-0)
  (:domain blocks-world)
  (:objects b0 b1 b2 b3 - block)
  (:init
    (ontable b0) (ontable b1) (ontable b2) (ontable b3) (arm-empty) (clear b0) (clear b1) (clear b2) (clear b3)
  )
  (:goal (and
    (on b0 b1) (ontable b1) (on b2 b3) (ontable b3)
  ))
)
