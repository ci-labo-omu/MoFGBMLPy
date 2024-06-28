cdef class RuleStyleSurvival:
    @staticmethod
    def replace(pop, offspring):
        # Sort by fitness (objective 0) # TODO: remove fitness class attribute in michigan solution and use objectives array instead

        # Check if we already exceed max num rules
        num_replacements = 0
        if args.get("MAX_NUM_RULE") < len(pop) + len(offspring): # TODO give this as param instead
            num_replacements = len(pop) + len(offspring) - args.get("MAX_NUM_RULE")


        # // Replace rules from bottom of list.
		# for(int i = 0; i < NumberOfReplacement; i++) {
		# 	currentList.set( (currentList.size()-1) - i , offspringList.get(i));
		# }
		# // Add rules
		# for(int i = NumberOfReplacement; i < offspringList.size(); i++) {
		# 	currentList.add(offspringList.get(i));
		# }

		# return pop

        raise Exception("Not yet implemented")