(define (problem blocksworld-large-35)
  (:domain blocksworld)
  (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b1) (on-table b10) (on-table b11) (on-table b15) (on-table b16) (on-table b6) (on b0 b13) (on b12 b3) (on b13 b4) (on b14 b9) (on b2 b0) (on b3 b2) (on b4 b7) (on b5 b1) (on b7 b11) (on b8 b14) (on b9 b15) (clear b10) (clear b12) (clear b16) (clear b5) (clear b6) (clear b8) (arm-empty)
  )
  (:goal
    (and (on-table b0) (on-table b1) (on b10 b8) (on b11 b9) (on b12 b10) (on b13 b11) (on b14 b12) (on b15 b13) (on b16 b14) (on b2 b0) (on b3 b1) (on b4 b2) (on b5 b3) (on b6 b4) (on b7 b5) (on b8 b6) (on b9 b7) (clear b15) (clear b16) (arm-empty))
  )
)
