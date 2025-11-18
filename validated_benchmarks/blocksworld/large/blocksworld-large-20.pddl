(define (problem blocksworld-large-20)
  (:domain blocksworld)
  (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b2 b20 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b0) (on-table b12) (on-table b16) (on-table b18) (on b1 b0) (on b10 b9) (on b11 b14) (on b13 b12) (on b14 b19) (on b15 b20) (on b17 b8) (on b19 b18) (on b2 b1) (on b20 b16) (on b3 b2) (on b4 b3) (on b5 b4) (on b6 b5) (on b7 b6) (on b8 b7) (on b9 b11) (clear b10) (clear b13) (clear b15) (clear b17) (arm-empty)
  )
  (:goal
    (and (on-table b0) (on b1 b0) (on b10 b9) (on b11 b10) (on b12 b11) (on b13 b12) (on b14 b13) (on b15 b14) (on b16 b15) (on b17 b16) (on b18 b17) (on b19 b18) (on b2 b1) (on b20 b19) (on b3 b2) (on b4 b3) (on b5 b4) (on b6 b5) (on b7 b6) (on b8 b7) (on b9 b8) (clear b20) (arm-empty))
  )
)
