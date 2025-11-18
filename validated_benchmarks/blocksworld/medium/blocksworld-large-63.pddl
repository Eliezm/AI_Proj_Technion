(define (problem blocksworld-large-63)
  (:domain blocksworld)
  (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b0) (on-table b12) (on-table b13) (on-table b15) (on-table b7) (on b1 b0) (on b10 b13) (on b11 b4) (on b14 b5) (on b16 b12) (on b2 b1) (on b3 b2) (on b4 b3) (on b5 b7) (on b6 b10) (on b8 b9) (on b9 b6) (clear b11) (clear b14) (clear b15) (clear b16) (clear b8) (arm-empty)
  )
  (:goal
    (and (on-table b0) (on b1 b0) (on b10 b9) (on b11 b10) (on b12 b11) (on b13 b12) (on b14 b13) (on b15 b14) (on b16 b15) (on b2 b1) (on b3 b2) (on b4 b3) (on b5 b4) (on b6 b5) (on b7 b6) (on b8 b7) (on b9 b8) (clear b16) (arm-empty))
  )
)
