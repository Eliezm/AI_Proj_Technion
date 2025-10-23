(define (problem blocks-world-4)
  (:domain blocks-world)
  (:objects b0 b1 b2 b3 b4 b5 b6 b7 - block)
  (:init
    (ontable b0) (ontable b1) (ontable b2) (ontable b3) (ontable b4) (ontable b5) (ontable b6) (ontable b7) (arm-empty) (clear b0) (clear b1) (clear b2) (clear b3) (clear b4) (clear b5) (clear b6) (clear b7)
  )
  (:goal (and
    (on b0 b1) (ontable b1) (on b2 b3) (ontable b3) (on b4 b5) (ontable b5) (ontable b6) (ontable b7)
  ))
)
