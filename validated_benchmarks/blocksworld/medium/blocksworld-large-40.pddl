(define (problem blocksworld-large-40)
  (:domain blocksworld)
  (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b0) (on-table b13) (on-table b14) (on-table b4) (on b1 b10) (on b10 b6) (on b11 b14) (on b12 b11) (on b15 b7) (on b16 b2) (on b2 b15) (on b3 b4) (on b5 b1) (on b6 b16) (on b7 b9) (on b8 b13) (on b9 b8) (clear b0) (clear b12) (clear b3) (clear b5) (arm-empty)
  )
  (:goal
    (and (on-table b11) (on-table b13) (on-table b16) (on-table b3) (on-table b9) (on b0 b3) (on b1 b6) (on b10 b16) (on b12 b10) (on b14 b2) (on b15 b13) (on b2 b12) (on b4 b5) (on b5 b1) (on b6 b7) (on b7 b14) (on b8 b11) (clear b0) (clear b15) (clear b4) (clear b8) (clear b9) (arm-empty))
  )
)
