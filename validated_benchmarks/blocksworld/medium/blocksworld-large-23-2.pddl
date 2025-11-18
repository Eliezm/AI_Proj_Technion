(define (problem blocksworld-large-23)
  (:domain blocksworld)
  (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b1) (on-table b10) (on-table b13) (on-table b2) (on b0 b10) (on b11 b5) (on b12 b9) (on b14 b3) (on b15 b16) (on b16 b12) (on b3 b8) (on b4 b6) (on b5 b0) (on b6 b2) (on b7 b15) (on b8 b13) (on b9 b11) (clear b1) (clear b14) (clear b4) (clear b7) (arm-empty)
  )
  (:goal
    (and (on-table b0) (on-table b1) (on-table b2) (on-table b3) (on b10 b6) (on b11 b7) (on b12 b8) (on b13 b9) (on b14 b10) (on b15 b11) (on b16 b12) (on b4 b0) (on b5 b1) (on b6 b2) (on b7 b3) (on b8 b4) (on b9 b5) (clear b13) (clear b14) (clear b15) (clear b16) (arm-empty))
  )
)
