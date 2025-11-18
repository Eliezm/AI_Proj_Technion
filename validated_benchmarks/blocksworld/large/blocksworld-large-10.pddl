(define (problem blocksworld-large-10)
  (:domain blocksworld)
  (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b2 b20 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b0) (on-table b11) (on-table b20) (on b1 b0) (on b10 b9) (on b12 b20) (on b13 b10) (on b14 b11) (on b15 b18) (on b16 b12) (on b17 b15) (on b18 b14) (on b19 b17) (on b2 b1) (on b3 b2) (on b4 b3) (on b5 b4) (on b6 b5) (on b7 b6) (on b8 b7) (on b9 b8) (clear b13) (clear b16) (clear b19) (arm-empty)
  )
  (:goal
    (and (on-table b0) (on b1 b0) (on b10 b9) (on b11 b10) (on b12 b11) (on b13 b12) (on b14 b13) (on b15 b14) (on b16 b15) (on b17 b16) (on b18 b17) (on b19 b18) (on b2 b1) (on b20 b19) (on b3 b2) (on b4 b3) (on b5 b4) (on b6 b5) (on b7 b6) (on b8 b7) (on b9 b8) (clear b20) (arm-empty))
  )
)
