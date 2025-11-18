(define (problem blocksworld-large-6)
  (:domain blocksworld)
  (:objects b0 b1 b10 b11 b12 b13 b14 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b13) (on-table b5) (on-table b9) (on b0 b1) (on b1 b2) (on b10 b8) (on b11 b6) (on b12 b10) (on b14 b13) (on b2 b7) (on b3 b12) (on b4 b5) (on b6 b3) (on b7 b4) (on b8 b9) (clear b0) (clear b11) (clear b14) (arm-empty)
  )
  (:goal
    (and (on-table b0) (on-table b1) (on b10 b8) (on b11 b9) (on b12 b10) (on b13 b11) (on b14 b12) (on b2 b0) (on b3 b1) (on b4 b2) (on b5 b3) (on b6 b4) (on b7 b5) (on b8 b6) (on b9 b7) (clear b13) (clear b14) (arm-empty))
  )
)
