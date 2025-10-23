(define (problem blocks-world-4)
  (:domain blocks-world)
  (:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 - block)
  (:init
    (ontable b0) (ontable b1) (ontable b2) (ontable b3) (ontable b4) (ontable b5) (ontable b6) (ontable b7) (ontable b8) (ontable b9) (ontable b10) (ontable b11) (arm-empty) (clear b0) (clear b1) (clear b2) (clear b3) (clear b4) (clear b5) (clear b6) (clear b7) (clear b8) (clear b9) (clear b10) (clear b11)
  )
  (:goal (and
    (on b0 b1) (on b1 b2) (ontable b2) (on b3 b4) (on b4 b5) (ontable b5) (on b6 b7) (on b7 b8) (ontable b8) (on b9 b10) (on b10 b11) (ontable b11)
  ))
)
